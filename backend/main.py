import json
from schemas.drink_request import DrinkRequest
from fastapi import FastAPI, HTTPException, Query, Request
import pandas as pd
import joblib

app = FastAPI()

# load model for score
model = joblib.load("model.pkl")
with open("model_features.json") as f:
    all_features = json.load(f)

# load text classification model and label
text_model = joblib.load("text_model.pkl")
with open("mlb_classes.json") as f:
    mlb_classes = json.load(f)

# load dataset
base_features = ["Calories", "Sugars (g)", "Protein (g)", "Total Fat (g)", "Caffeine (mg)", "Calcium (% DV)"]
df = pd.read_csv("labeled_starbucks.csv")
df.columns = [col.strip() for col in df.columns]

def preprocess_text(text):
    text = text.lower()
    synonyms = {
        "calories": "calorie",
        "low calorie": "low_calorie",
        "low-calorie": "low_calorie"
    }
    for key, val in synonyms.items():
        text = text.replace(key, val)
    return text

def compute_score(row: pd.Series) -> float:
    X_base = row[base_features].to_dict()
    X_dummy = {col: 0 for col in all_features if col not in base_features}

    size_col = f"Size_{row['Size']}"
    milk_col = f"Milk_Type_{row['Milk_Type']}"
    whipped_col = f"Whipped_Cream_{row['Whipped_Cream']}"

    for col in [size_col, milk_col, whipped_col]:
        if col in X_dummy:
            X_dummy[col] = 1

    X_full = {**X_base, **X_dummy}
    X_df = pd.DataFrame([X_full])[all_features]
    return model.predict(X_df)[0]

def get_drink_row(request) -> pd.Series:
    filtered = df[
        (df["Beverage"] == request.beverage) &
        (df["Size"] == request.size) &
        (df["Milk_Type"] == request.milk) &
        (df["Whipped_Cream"] == request.whipped_cream)
    ]
    return filtered.iloc[0] if not filtered.empty else None

def find_better_suggestions(original_score: float, request) -> list:
    suggestions = []
    same_beverage_df = df[df["Beverage"] == request.beverage].copy()
    same_beverage_df["pred_score"] = same_beverage_df.apply(compute_score, axis=1)
    better_df = same_beverage_df[same_beverage_df["pred_score"] > original_score]
    if better_df.empty:
        return []

    better_df["score_diff"] = better_df["pred_score"] - original_score
    top3 = better_df.sort_values("score_diff").head(3)
    for _, row_sug in top3.iterrows():
        suggestions.append({
            "beverage": row_sug["Beverage"],
            "size": row_sug["Size"],
            "milk": row_sug["Milk_Type"],
            "whipped_cream": bool(row_sug["Whipped_Cream"]),
            "score": round(row_sug["pred_score"], 2)
        })
    return suggestions

def suggest_healthy_drinks_by_category(category: str, top_n: int = 3) -> list:
    if "Beverage_category" not in df.columns:
        raise ValueError("Missing 'Beverage_category' column in dataset")

    candidates = df[df["Beverage_category"] == category].copy()
    if candidates.empty:
        return []

    candidates["pred_score"] = candidates.apply(compute_score, axis=1)
    top_df = candidates.sort_values("pred_score", ascending=False).head(top_n)
    results = []
    for _, row in top_df.iterrows():
        results.append({
            "beverage": row["Beverage"],
            "size": row["Size"],
            "milk": row["Milk_Type"],
            "whipped_cream": bool(row["Whipped_Cream"]),
            "score": round(row["pred_score"], 2)
        })
    return results


@app.post("/predict")
def predict_score(request: DrinkRequest):
    row = get_drink_row(request)
    if row is None:
        raise HTTPException(status_code=404, detail="Cannot find the drink")

    original_score = compute_score(row)

    if original_score >= 80:
        message = "✅ Good after workout!"
    elif original_score >= 60:
        message = "🤔 Hmm... maybe okay!"
    else:
        message = "🍰 Save it for cheat day!"

    suggestions = find_better_suggestions(original_score, request) if original_score <= 60 else None

    response = {
        "score": round(original_score, 2),
        "message": message,
        "input": request.model_dump(),
        "suggestions": suggestions if suggestions else None
    }

    if not suggestions:
        response["note"] = "😕 No better version of this drink was found. Try choosing a different base drink!"

    return response


@app.get("/suggest_alternatives")
def suggest_alternatives(category: str = Query(...), top_n: int = Query(3)):
    try:
        top_drinks = suggest_healthy_drinks_by_category(category, top_n)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not top_drinks:
        raise HTTPException(status_code=404, detail="No drinks found in that category.")

    return {
        "category": category,
        "top_drinks": top_drinks
    }

@app.get("/recommend_text")
async def recommend_from_text(request: Request):
    data = await request.json()
    user_input = request.get("text", "")
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing input text")

    # step 1: predict tag(s)
    probs = text_model.predict_proba([preprocess_text(user_input)])[0]
    threshold = 0.3
    predict_indices = [i for i, p in enumerate(probs) if p >= threshold]
    predicted_tags = [mlb_classes[i] for i in predict_indices]
    if not predicted_tags:
        predicted_tags = ["none"]

    # step 2: search drinks that match the tag(s)
    matched = df[df["tags"].apply(lambda row_tags: all(tag in row_tags for tag in predicted_tags))] if "tags" in df.columns else pd.DataFrame()

    if matched.empty:
        partial = df[df["tags"].apply(lambda row_tags: any(
            tag in row_tags for tag in predicted_tags))] if "tags" in df.columns else pd.DataFrame()
        top = partial.sort_values(by="score", ascending=False)[["Beverage", "tags", "score"]].head(5)
        message = "No exact match found. Showing partial matches instead."
    else:
        top = matched.sort_values(by="score", ascending=False)[["Beverage", "tags", "score"]].head(5)
        message = "Here are your best matches!"

    return {
        "input_text": user_input,
        "predicted_tags": predicted_tags,
        "message": message,
        "recommendations": top.to_dict(orient="records")
    }
