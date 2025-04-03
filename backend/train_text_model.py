import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import joblib


# -------- 1. training data --------------
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

    ("I want something sweet and high in protein", ["sweet", "high_protein"]),
    ("Low-calorie but also caffeinated", ["low_calorie", "caffeine_boost"]),
    ("A sweet, low-calorie drink for me", ["sweet", "low_calorie"]),
]

texts = [t[0] for t in training_data]
label_lists = [t[1] for t in training_data]

# ---------- 2. Multi-Label binarization ----------------
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_lists)

# ---------- 3. Text processing -------------------------
def process_text(text):
    text = text.lower()
    synonyms = {
        "calories": "calorie",
        "low calorie": "low_calorie",
        "low-calorie": "low_calorie"
    }
    for key, val in synonyms.items():
        text = text.replace(key, val)
    return text

# --------- 4. Build model pipe line --------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000)))
])
model.fit(texts, Y)

# --------- 5. Save model and label classes -------------
joblib.dump(model, "text_model.pkl")
print("Model save as text_model.pkl")

with open("mlb_classes.json", "w") as f:
    json.dump(mlb.classes_.tolist(), f)
print("Label classes saved as mlb_classes.json")