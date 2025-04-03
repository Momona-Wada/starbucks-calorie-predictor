import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from schemas.drink_request import DrinkRequest
from schemas.recommend_text import RecommendTextRequest
from text_preprocessor import preprocess_text

app = FastAPI()

# === load model for score
model = joblib.load("model.pkl")
with open("model_features.json") as f:
    all_features = json.load(f)

# === load text classification model
text_model = joblib.load("text_model.pkl")
with open("mlb_classes.json") as f:
    mlb_classes = json.load(f)

# === load dataset
base_features = ["Calories", "Sugars (g)", "Protein (g)", "Total Fat (g)", "Caffeine (mg)", "Calcium (% DV)"]

df = pd.read_csv("labeled_starbucks_with_tags.csv")

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

    # ... 省略せずに貼る
    same_beverage_df = df[df["Beverage"] == request.beverage].copy()
    same_beverage_df["pred_score"] = same_beverage_df.apply(compute_score, axis=1)
    better_df = same_beverage_df[same_beverage_df["pred_score"] > original_score]
    if not better_df.empty:
        better_df["score_diff"] = better_df["pred_score"] - original_score
        top3 = better_df.sort_values("score_diff").head(3)
        suggestions = []
        for _, row_sug in top3.iterrows():
            suggestions.append({
                "beverage": row_sug["Beverage"],
                "size": row_sug["Size"],
                "milk": row_sug["Milk_Type"],
                "whipped_cream": bool(row_sug["Whipped_Cream"]),
                "score": round(row_sug["pred_score"], 2)
            })
    else:
        suggestions = None

    response = {
        "score": round(original_score, 2),
        "message": message,
        "input": request.model_dump(),
        "suggestions": suggestions
    }
    if not suggestions:
        response["note"] = "😕 No better version of this drink was found."
    return response

@app.post("/recommend_text")
def recommend_from_text(body: RecommendTextRequest):
    user_input = body.text
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing input text")

    # step1: predict tags
    probs = text_model.predict_proba([preprocess_text(user_input)])[0]
    threshold = 0.3
    predicted_indices = [i for i, p in enumerate(probs) if p >= threshold]
    predicted_tags = [mlb_classes[i] for i in predicted_indices]
    if not predicted_tags:
        predicted_tags = ["none"]

    # step2: filter drinks with these tags
    if "tags" not in df.columns:
        # user forgot to generate tags -> error
        raise HTTPException(status_code=500, detail="No 'tags' column found. Please generate tags first.")

    matched = df[df["tags"].apply(lambda row_tags: all(tag in row_tags for tag in predicted_tags))]

    if matched.empty:
        partial = df[df["tags"].apply(lambda row_tags: any(tag in row_tags for tag in predicted_tags))]
        if partial.empty:
            return {
                "input_text": user_input,
                "predicted_tags": predicted_tags,
                "message": "No drinks found at all for these tags",
                "recommendations": []
            }
        if "score" not in partial.columns:
            partial["score"] = partial.apply(compute_score, axis=1)
        top = partial.sort_values(by="score", ascending=False)[["Beverage", "tags", "score"]].head(5)
        message = "No exact match found. Showing partial matches instead."
    else:
        if "score" not in matched.columns:
            matched["score"] = matched.apply(compute_score, axis=1)
        top = matched.sort_values(by="score", ascending=False)[["Beverage", "tags", "score"]].head(5)
        message = "Here are your best matches!"

    return {
        "input_text": user_input,
        "predicted_tags": predicted_tags,
        "message": message,
        "recommendations": top.to_dict(orient="records")
    }
