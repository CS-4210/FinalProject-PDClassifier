import joblib
import pandas as pd

# saved model pipeline produced by train.py
MODEL_PATH = "model/parkinsons_model.pkl"
# saved list of feature cols in the exact order expected by the model
FEATURE_PATH = "model/feature_columns.pkl"

# loads the saved trained model + feature list from disk
def load_model():
    # load the trained sklearn pipeline
    model = joblib.load(MODEL_PATH)
    # load the exact feature order used during training
    feature_columns = joblib.load(FEATURE_PATH)
    return model, feature_columns


# conversta feature dictionary into a PD prediction
def predict_parkinsons(input_dict):
    model, feature_columns = load_model()

    # convert user input into a one row DF
    input_df = pd.DataFrame([input_dict])
    # reorder cols to match the order used during training
    input_df = input_df.reindex(columns=feature_columns)

    # predict teh class label
    prediction = model.predict(input_df)[0]
    # get teh probability of PD
    probability = model.predict_proba(input_df)[0][1]

    # convert numeric class into readble label
    label = "Parkinson's Disease" if prediction == 1 else "Healthy"

    return {
        "prediction": int(prediction),
        "label": label,
        "probability": float(probability)
    }