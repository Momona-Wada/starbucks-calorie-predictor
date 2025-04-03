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
    ("I want something sweet", ["sweet"]),
    ("I'm craving sugar", ["sweet"]),
    ("Give me a big sugary drink", ["sweet"]),
    ("not sweet at all", ["not_sweet"]),
    ("I need something low in sugar", ["not_sweet"]),

    ("I'm looking for a low-calorie drink", ["low_calorie"]),
    ("I'm looking for a low calories drink", ["low_calorie"]),
    ("Something light and healthy", ["low_calorie"]),
    ("I'm on a diet, so low calorie please", ["low_calorie"]),
    ("A drink with not many calories", ["low_calorie"]),

    ("High protein for workout recovery", ["high_protein"]),
    ("Protein-packed beverage", ["high_protein"]),
    ("I want a drink with lots of protein", ["high_protein"]),

    ("Give me a caffeine boost", ["caffeine_boost"]),
    ("I need a strong coffee", ["caffeine_boost"]),
    ("I want something energizing", ["caffeine_boost"]),

    ("not sweet", ["not_sweet"]),
    ("no sugar please", ["low_sugar"]),
    ("avoid caffeine", ["no_caffeine"]),
    ("high calorie", ["high_calorie"]),
    ("very high calorie", ["high_calorie", "unhealthy"]),
    ("I want something unhealthy", ["unhealthy"]),
    ("not a lot of protein", []),
    ("no milk", ["non_dairy"]),
    ("I want dessert in a cup", ["dessert_like", "sweet"]),

    ("I want low sugar", ["low_sugar"]),
    ("no sugar please", ["low_sugar"]),
    ("I don’t want a sugary drink", ["low_sugar"]),
    ("please give me something low in sugar", ["low_sugar"]),
    ("I’m trying to cut back on sugar", ["low_sugar"]),
    ("a drink with no sugar", ["low_sugar"]),
    ("not much sugar", ["low_sugar"]),
    ("less sugar", ["low_sugar"]),
    ("sugar-free", ["low_sugar"]),
    ("I want a drink with minimum sugar", ["low_sugar"]),

    ("no caffeine", ["no_caffeine"]),
    ("I don't want caffeine", ["no_caffeine"]),
    ("zero caffeine please", ["no_caffeine"]),
    ("decaf only", ["no_caffeine"]),
    ("I’m avoiding caffeine", ["no_caffeine"]),
    ("non-caffeinated drink", ["no_caffeine"]),
    ("I prefer drinks without caffeine", ["no_caffeine"]),
    ("no energy boost", ["no_caffeine"]),
    ("caffeine-free", ["no_caffeine"]),
    ("I want something calming", ["no_caffeine"]),

    ("high protein", ["high_protein"]),
    ("I want something high in protein", ["high_protein"]),
    ("something protein-rich", ["high_protein"]),
    ("a drink with lots of protein", ["high_protein"]),
    ("give me protein", ["high_protein"]),
    ("I need a protein boost", ["high_protein"]),
    ("protein-packed", ["high_protein"]),
    ("for muscle recovery", ["high_protein"]),
    ("good for after workout", ["high_protein"]),
    ("something that helps recovery", ["high_protein"]),
    ("post-workout drink", ["high_protein"]),
    ("protein-heavy drink", ["high_protein"]),
    ("extra protein please", ["high_protein"]),
    ("lots of protein", ["high_protein"]),
    ("build muscle drink", ["high_protein"]),
    ("something gym-friendly", ["high_protein"]),
    ("drink for fitness", ["high_protein"]),
    ("fuel for muscles", ["high_protein"]),
    ("protein-filled beverage", ["high_protein"]),
    ("nutritious and full of protein", ["high_protein"]),

    ("low protein", ["low_protein"]),
    ("I want something low in protein", ["low_protein"]),
    ("not a lot of protein", ["low_protein"]),
    ("avoid protein", ["low_protein"]),
    ("less protein please", ["low_protein"]),
    ("I don't need protein", ["low_protein"]),
    ("not high in protein", ["low_protein"]),
    ("I'm avoiding protein", ["low_protein"]),
    ("light protein drink", ["low_protein"]),
    ("minimal protein", ["low_protein"]),
    ("no protein boost", ["low_protein"]),
    ("drink with little protein", ["low_protein"]),
    ("something not protein-heavy", ["low_protein"]),
    ("protein-free drink", ["low_protein"]),
    ("no extra protein", ["low_protein"]),

    ("low calorie", ["low_calorie"]),
    ("I'm looking for something low in calories", ["low_calorie"]),
    ("less calories please", ["low_calorie"]),
    ("I need a light drink", ["low_calorie"]),
    ("not a high calorie drink", ["low_calorie"]),
    ("something diet friendly", ["low_calorie"]),
    ("low cal", ["low_calorie"]),
    ("drink with fewer calories", ["low_calorie"]),
    ("healthy and low calorie", ["low_calorie"]),
    ("minimum calorie drink", ["low_calorie"]),
    ("I'm watching my calories", ["low_calorie"]),
    ("I prefer low calorie drinks", ["low_calorie"]),
    ("calorie-conscious choice", ["low_calorie"]),
    ("nothing heavy please", ["low_calorie"]),
    ("I’m on a diet", ["low_calorie"]),

    ("I want dessert in a cup", ["dessert_like"]),
    ("Like a dessert drink", ["dessert_like"]),
    ("Make it taste like dessert", ["dessert_like"]),
    ("A drink that feels like a dessert", ["dessert_like"]),
    ("Something creamy and rich like dessert", ["dessert_like"]),
    ("I want a drink that could be a dessert", ["dessert_like"]),
    ("Give me something decadent", ["dessert_like"]),
    ("Make it feel like a treat", ["dessert_like"]),
    ("I’m craving something dessert-y", ["dessert_like"]),
    ("I want a drink that’s basically a cake", ["dessert_like"]),

    # Multi-label
    ("I want something sweet and high in protein", ["sweet", "high_protein"]),
    ("Low-calorie but also caffeinated", ["low_calorie", "caffeine_boost"]),
    ("A sweet, low-calorie drink for me", ["sweet", "low_calorie"]),
    ("It's my cheat day", ["high_calorie", "sweet"]),
    ("Treat myself with something sweet", ["high_calorie", "sweet"]),
    ("I want a high calorie dessert drink", ["high_calorie", "sweet"]),
    ("Give me something indulgent", ["high_calorie", "sweet"]),
    ("Something super sweet and rich", ["high_calorie", "sweet"]),
    ("Today is my cheat day", ["high_calorie", "sweet"]),
    ("I don’t care about calories today", ["high_calorie", "sweet"]),
    ("A sugary high calorie treat", ["high_calorie", "sweet"]),
    ("I deserve something sweet and heavy", ["high_calorie", "sweet"]),
    ("I want a cheat day treat", ["high_calorie", "sweet"]),
    ("Go big on sugar and calories", ["high_calorie", "sweet"]),
    ("Cheat treat time!", ["high_calorie", "sweet"]),

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
