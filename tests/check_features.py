import joblib

features = joblib.load("model/feature_columns.pkl")

print(features)
print("Number of features:", len(features))
print("Contains name?", "name" in features)