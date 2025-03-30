import re

import pandas as pd

def parse_beverage_prep(bp):
    sizes = ["Short", "Tall", "Grande", "Venti"]
    bp = str(bp).strip()
    size = None
    milk = bp
    for s in sizes:
        if bp.startswith(s):
            size = s
            milk = bp[len(s):].strip()
            break
    return size, milk

def parse_whipped_cream(beverage):
    beverage = str(beverage)
    if "(" in beverage and ")" in beverage:
        if "Without Whipped Cream" in beverage:
            return False
        elif "With Whipped Cream" in beverage:
            return True
    return False

def clean_beverage_name(beverage):
    return re.sub(r"\s*\(.*Whipped Cream.*\)", "", beverage, flags=re.IGNORECASE).strip()

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

    if row.get("Whipped_Cream", False):
        score -= 5

    milk = str(row.get("Milk_Type", "")).lower()
    if "soy" in milk:
        score += 3
    elif "nonfat" in milk:
        score += 3
    elif "2%" in milk:
        score += 1
    elif "whole" in milk:
        score -= 3

    size = row.get("Size", "")
    if size == "Venti":
        score -= 5
    elif size == "Grande":
        score -= 3
    elif size == "Short":
        score += 2

    score = max(0, min(100, score)) # 0 - 100
    return score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

df = pd.read_csv("starbucks.csv")
df.columns = [col.strip() for col in df.columns]

# create 'Size' and 'Milk_Type' columns from 'Beverage_prep'
df["Size"], df["Milk_Type"] = zip(*df["Beverage_prep"].apply(parse_beverage_prep))

# create 'Whipped_Cream' column from 'Beverage'
df["Whipped_Cream"] = df["Beverage"].apply(parse_whipped_cream)

# remove whipped cream info from 'Beverage'
df["Beverage"] = df["Beverage"].apply(clean_beverage_name)

numeric_cols = ["Calories", "Sugars (g)", "Protein (g)", "Total Fat (g)", "Caffeine (mg)", "Calcium (% DV)"]
for col in numeric_cols:
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


