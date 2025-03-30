import json
from schemas.drink_request import DrinkRequest
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
with open("model_features.json") as f:
    all_features = json.load(f)

base_features = ["Calories", "Sugars (g)", "Protein (g)", "Total Fat (g)", "Caffeine (mg)", "Calcium (% DV)"]
df = pd.read_csv("labeled_starbucks.csv")
df.columns = [col.strip() for col in df.columns]


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