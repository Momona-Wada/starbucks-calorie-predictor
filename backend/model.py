import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
df = pd.read_csv("labeled_starbucks.csv")

categorical_cols = ["Size", "Milk_Type", "Whipped_Cream"]
df_encoded = pd.get_dummies(df, columns=categorical_cols)

base_features = ["Calories", "Sugars (g)", "Protein (g)", "Total Fat (g)", "Caffeine (mg)", "Calcium (% DV)"]
dummy_features = [col for col in df_encoded.columns if col.startswith(("Size_", "Milk_Type_", "Whipped_Cream_"))]

features = base_features + dummy_features
X = df_encoded[features]
y = df_encoded["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_absolute_error(y_test, y_pred)

print(f"R² score: {r2}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")

joblib.dump(model, "model.pkl")
print(f"Model is successfully saved to model.pkl")

