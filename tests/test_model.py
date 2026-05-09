import pandas as pd
import joblib

df = pd.read_csv("data/parkinsons.data")

if "name" in df.columns:
    df = df.drop(columns=["name"])

X = df.drop(columns=["status"])
y = df["status"]

model = joblib.load("model/parkinsons_model.pkl")
features = joblib.load("model/feature_columns.pkl")

healthy_sample = X[y == 0].iloc[0]
pd_sample = X[y == 1].iloc[0]

print("Healthy prediction:")
print(model.predict(pd.DataFrame([healthy_sample]).reindex(columns=features)))
print(model.predict_proba(pd.DataFrame([healthy_sample]).reindex(columns=features)))

print("\nParkinson's prediction:")
print(model.predict(pd.DataFrame([pd_sample]).reindex(columns=features)))
print(model.predict_proba(pd.DataFrame([pd_sample]).reindex(columns=features)))