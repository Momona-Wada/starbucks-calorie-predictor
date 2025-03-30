from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
df = pd.read_csv("labeled_starbucks.csv")
df.columns = [col.strip() for col in df.columns]

class DrinkRequest(BaseModel):
    beverage: str
    prep: str

@app.post("/predict")
def predict_score(request: DrinkRequest):
    row = df[
        (df["Beverage"] == request.beverage) & (df["Beverage_prep"] == request.prep)
    ]
    if row.empty:
        raise HTTPException(status_code=404, detail="Cannot find the drink")

    features = ["Calories", "Sugars (g)", "Protein (g)", "Total Fat (g)", "Caffeine (mg)", "Calcium (% DV)"]
    X = row[features]
    score = model.predict(X)[0]

    if score >= 80:
        message = "Good after workout!"
    elif score >= 60:
        message = "hmmm Ok, but be careful!"
    else:
        message = "Cheat day only!"

    return {
        "score": round(score, 2),
        "message": message,
        "beverage": request.beverage,
        "prep": request.prep
    }