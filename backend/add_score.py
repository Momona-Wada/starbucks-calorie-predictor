import pandas as pd

def calculate_score(row):
    score = 100

    protein = row["Protein (g)"]
    if protein >= 10:
        score += 20
    elif protein >= 6:
        score += 10

    sugars = row["Sugars (g)"]
    if sugars >= 30:
        score -= 30
    elif sugars >= 20:
        score -= 20
    elif sugars >= 10:
        score -= 10

    calories = row["Calories"]
    if calories >= 300:
        score -= 20
    elif calories >= 200:
        score -= 10

    fat = row["Total Fat (g)"]
    if fat >= 10:
        score -= 10

    caffeine = row["Caffeine (mg)"]
    if caffeine >= 350:
        score -= 10
    elif caffeine <= 50:
        score += 5

    calcium = row["Calcium (% DV)"]
    if isinstance(calcium, str):
        calcium = calcium.replace("%", "")
    try:
        calcium = float(calcium)
    except:
        calcium = 0.0

    if calcium >= 20:
        score += 10
    elif calcium >= 10:
        score += 5

    score = max(0, min(100, score)) # 0 - 100
    return score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

df = pd.read_csv("starbucks.csv")

df.columns = [col.strip() for col in df.columns]

for col in ["Calories", "Sugars (g)", "Protein (g)", "Total Fat (g)", "Caffeine (mg)", "Calcium (% DV)"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("Varies", "0", regex=False)
        .str.replace("varies", "0", regex=False)
        .str.replace("nan", "0", regex=False)
    )
    df[col] = pd.to_numeric(df[col]).fillna(0)

df["score"] = df.apply(calculate_score, axis=1)
df.to_csv("labeled_starbucks.csv", index=False)


