import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

# === Step 1: Load Starbucks drink data ===
df = pd.read_csv("labeled_starbucks.csv")

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

# === Optional: Synonym expansion function ===
# Simple example that replaces certain words with synonyms or standard forms.
# In a real application, you might use a more sophisticated approach (e.g., WordNet).
def preprocess_text(text):
    text = text.lower()
    # Replace some synonyms or variations with common tokens
    synonyms = {
        "calories": "calorie",  # singular form
        "low calorie": "low_calorie",  # unify with underscore
        "low-calorie": "low_calorie"
        # Add more synonyms/variations if needed
    }
    for key, val in synonyms.items():
        text = text.replace(key, val)
    return text

# === Step 3: Multi-label training data for text → tags prediction ===
# Include variations like "low calories" for better coverage.
training_data = [
    ("I want something sweet", ["sweet"]),
    ("I'm craving sugar", ["sweet"]),
    ("Give me a big sugary drink", ["sweet"]),
    ("not sweet at all", []),  # negative example
    ("I need something low in sugar", []),  # negative example

    ("I'm looking for a low-calorie drink", ["low_calorie"]),
    ("I'm looking for a low calories drink", ["low_calorie"]),  # new phrase
    ("Something light and healthy", ["low_calorie"]),
    ("I'm on a diet, so low calorie please", ["low_calorie"]),
    ("A drink with not many calories", ["low_calorie"]),        # new phrase

    ("High protein for workout recovery", ["high_protein"]),
    ("Protein-packed beverage", ["high_protein"]),
    ("I want a drink with lots of protein", ["high_protein"]),

    ("Give me a caffeine boost", ["caffeine_boost"]),
    ("I need a strong coffee", ["caffeine_boost"]),
    ("I want something energizing", ["caffeine_boost"]),

    # Multi-label examples:
    ("I want something sweet and high in protein", ["sweet", "high_protein"]),
    ("Low-calorie but also caffeinated", ["low_calorie", "caffeine_boost"]),
    ("A sweet, low-calorie drink for me", ["sweet", "low_calorie"]),
]

texts = [t[0] for t in training_data]
label_lists = [t[1] for t in training_data]

# Convert list of labels into a multi-label binary matrix
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_lists)

# === Step 4: Train the multi-label classification model ===
model = Pipeline([
    # Preprocess text for synonyms, etc.
    ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1,2))),
    ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
])
model.fit(texts, Y)

# === Step 5: Use the model to recommend drinks ===
def recommend_drinks(user_input):
    # Preprocess user input to handle synonyms or variations
    user_input_processed = preprocess_text(user_input)

    # Predict probabilities for each label
    probs = model.predict_proba([user_input_processed])[0]

    # Lower threshold to, for example, 0.3 to catch borderline cases
    threshold = 0.3
    predicted_indices = [i for i, p in enumerate(probs) if p >= threshold]
    predicted_tags = [mlb.classes_[i] for i in predicted_indices]

    # If no tags predicted, we can treat it as "none" or handle gracefully
    if not predicted_tags:
        predicted_tags = ["none"]
    print(f"[Predicted tags] → {predicted_tags}")

    # Filter drinks that match *all* predicted tags (strict intersection)
    matched = df[df["tags"].apply(lambda row_tags: all(tag in row_tags for tag in predicted_tags))]

    if matched.empty:
        print("No matching drinks found. Show partial matches:")
        partial = df[df["tags"].apply(lambda row_tags: any(tag in row_tags for tag in predicted_tags))]
        top = partial.sort_values(by="score", ascending=False)[["Beverage", "tags", "score"]].head()
        print(top.to_string(index=False))
    else:
        # Sort matched DataFrame by 'score' or other priority
        top = matched.sort_values(by="score", ascending=False)[["Beverage", "tags", "score"]].head()
        print("[Top 5 Recommended Drinks]")
        print(top.to_string(index=False))

# === Run the program ===
if __name__ == "__main__":
    print("☕ AI-Powered Starbucks Recommender ☕")
    user_text = input("What kind of drink are you craving today? > ")
    recommend_drinks(user_text)
