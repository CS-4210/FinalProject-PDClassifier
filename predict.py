import joblib
import pandas as pd


MODEL_PATH = "model/parkinsons_model.pkl"
FEATURE_PATH = "model/feature_columns.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    return model, feature_columns


def predict_parkinsons(input_dict):
    model, feature_columns = load_model()

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    label = "Parkinson's Disease" if prediction == 1 else "Healthy"

    return {
        "prediction": int(prediction),
        "label": label,
        "probability": float(probability)
    }