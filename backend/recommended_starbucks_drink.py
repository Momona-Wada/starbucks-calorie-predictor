import pandas as pd
import joblib
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from text_preprocessor import preprocess_text

# === Step 1: Load Starbucks drink data ===
df = pd.read_csv("labeled_starbucks.csv")

# === Step 2: Generate nutrition-based tags for each drink ===
def generate_tags(row):
    tags = []
    if row['Sugars (g)'] > 30:
        tags.append("sweet")
    elif row['Sugars (g)'] < 10:
        tags.append("low_sugar")
    else:
        tags.append("not_sweet")

    if row['Calories'] < 100:
        tags.append("low_calorie")
    elif row['Calories'] > 400:
        tags.append("high_calorie")
        tags.append("unhealthy")

    if row['Protein (g)'] > 10:
        tags.append("high_protein")

    if row['Caffeine (mg)'] > 150:
        tags.append("caffeine_boost")
    elif row['Caffeine (mg)'] < 10:
        tags.append("no_caffeine")

    milk = row.get("Mild_Type", "")
    if isinstance(milk, str) and ("Almond" in milk or "Oat" in milk):
        tags.append("non_dairy")

    return tags if tags else ["none"]

df["tags"] = df.apply(generate_tags, axis=1)

df.to_csv("labeled_starbucks_with_tags.csv", index=False)

# === Step 3: Multi-label training data for text → tags prediction
training_data = [
    # --- sweet ---
    ("I want something sweet", ["sweet"]),
    ("Give me a sugary drink", ["sweet"]),
    ("A dessert-like sweet beverage", ["sweet"]),

    # --- not_sweet ---
    ("No sweetness at all", ["not_sweet"]),
    ("I want something bitter", ["not_sweet"]),
    ("Nothing sweet, just plain", ["not_sweet"]),

    # --- low_sugar ---
    ("I want low sugar", ["low_sugar"]),
    ("no sugar please", ["low_sugar"]),
    ("less sugar", ["low_sugar"]),

    # --- low_calorie ---
    ("I want something low in calories", ["low_calorie"]),
    ("I’m on a diet, so low calorie please", ["low_calorie"]),
    ("healthy and low calorie", ["low_calorie"]),

    # --- high_calorie ---
    ("Give me a high calorie drink", ["high_calorie"]),
    ("I want something heavy", ["high_calorie"]),
    ("lots of calories", ["high_calorie"]),

    # --- unhealthy ---
    ("I want something super unhealthy", ["unhealthy"]),
    ("Forget the health stuff, I want sugar", ["unhealthy"]),
    ("I want a sugary bomb", ["unhealthy"]),

    # --- high_protein ---
    ("I want high protein", ["high_protein"]),
    ("I need a protein boost", ["high_protein"]),
    ("a drink with lots of protein", ["high_protein"]),

    # --- no_caffeine ---
    ("no caffeine", ["no_caffeine"]),
    ("decaf only", ["no_caffeine"]),
    ("caffeine-free", ["no_caffeine"]),
    ("decaf only", ["no_caffeine"]),
    ("I want decaf coffee", ["no_caffeine"]),
    ("decaf, please", ["no_caffeine"]),
    ("Give me a decaf drink", ["no_caffeine"]),
    ("decaf", ["no_caffeine"]),

    # --- caffeine_boost ---
    ("caffeine boost", ["caffeine_boost"]),
    ("I want a strong coffee", ["caffeine_boost"]),
    ("lots of caffeine", ["caffeine_boost"]),
    ("I need a caffeine hit", ["caffeine_boost"]),
    ("Give me a caffeine kick", ["caffeine_boost"]),
    ("I need my caffeine fix", ["caffeine_boost"]),
    ("I want a drink loaded with caffeine", ["caffeine_boost"]),
    ("I crave a highly caffeinated drink", ["caffeine_boost"]),
    ("I need an energy boost", ["caffeine_boost"]),
    ("Hit me with some caffeine", ["caffeine_boost"]),
    ("I want a caffeine explosion", ["caffeine_boost"]),
    ("I want a drink to wake me up", ["caffeine_boost"]),

    # --- non_dairy ---
    ("no milk", ["non_dairy"]),
    ("Use almond milk please", ["non_dairy"]),
    ("I prefer oat milk", ["non_dairy"]),

    # --- dessert_like ---
    ("I want dessert in a cup", ["dessert_like"]),
    ("Make it taste like dessert", ["dessert_like"]),
    ("A drink that feels like a dessert", ["dessert_like"]),

    # --- low_protein ---
    ("I want something low in protein", ["low_protein"]),
    ("avoid protein", ["low_protein"]),
    ("not high in protein", ["low_protein"]),

    # --- multi-label examples ---
    ("I want something sweet and high in protein", ["sweet", "high_protein"]),
    ("I want sweet and low calorie", ["sweet", "low_calorie"]),
    ("I want sweet but no caffeine", ["sweet", "no_caffeine"]),
    ("I want something sweet and unhealthy", ["sweet", "unhealthy"]),
    ("I want a dessert but low sugar", ["dessert_like", "low_sugar"]),
    ("I want a high calorie dessert", ["dessert_like", "high_calorie"]),
]
texts = [t[0] for t in training_data]
label_lists = [t[1] for t in training_data]

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_lists)

model = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1,2))),
    ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
])

model.fit(texts, Y)

joblib.dump(model, "text_model.pkl")
with open("mlb_classes.json", "w") as f:
    json.dump(mlb.classes_.tolist(), f)

print("Saved text_model.pkl, mlb_classes.json, and labeled_starbucks_with_tags.csv!")
