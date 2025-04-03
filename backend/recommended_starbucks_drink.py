# recommended_starbucks_drink.py
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
df = pd.read_csv("labeled_starbucks.csv")  # ここがタグなしのCSVと想定

# === Step 2: Generate nutrition-based tags for each drink ===
def generate_tags(row):
    tags = []
    if row['Sugars (g)'] > 30:
        tags.append("sweet")
    if row['Calories'] < 100:
        tags.append("low_calorie")
    if row['Protein (g)'] > 10:
        tags.append("high_protein")
    if row['Caffeine (mg)'] > 150:
        tags.append("caffeine_boost")
    return tags if tags else ["none"]

df["tags"] = df.apply(generate_tags, axis=1)

# ここでtags付きCSVを保存
df.to_csv("labeled_starbucks_with_tags.csv", index=False)

# === Step 3: Multi-label training data for text → tags prediction
training_data = [
    ("I want something sweet", ["sweet"]),
    ("I'm craving sugar", ["sweet"]),
    ("Give me a big sugary drink", ["sweet"]),
    ("not sweet at all", []),
    ("I need something low in sugar", []),

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

    # Multi-label
    ("I want something sweet and high in protein", ["sweet", "high_protein"]),
    ("Low-calorie but also caffeinated", ["low_calorie", "caffeine_boost"]),
    ("A sweet, low-calorie drink for me", ["sweet", "low_calorie"]),
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

# モデルとマルチラベルの情報を出力
joblib.dump(model, "text_model.pkl")
with open("mlb_classes.json", "w") as f:
    json.dump(mlb.classes_.tolist(), f)

print("Saved text_model.pkl, mlb_classes.json, and labeled_starbucks_with_tags.csv!")
