import streamlit as st
import pandas as pd
import joblib
import numpy as np


MODEL_PATH = "model/parkinsons_model.pkl"
FEATURE_PATH = "model/feature_columns.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    return model, feature_columns


def predict(model, feature_columns, user_input):
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=feature_columns)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability, input_df


st.set_page_config(
    page_title="Parkinson's Voice Classifier",
    page_icon="🧠",
    layout="wide"
)

model, feature_columns = load_artifacts()

st.title("Parkinson's Disease Voice Classifier")
st.caption("Machine learning prototype using biomedical voice features")

st.warning(
    "This app is for educational demonstration only. It is not a medical diagnosis tool."
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Project Motivation")
    st.write(
        """
        This project predicts Parkinson's Disease using voice measurements such as 
        fundamental frequency, jitter, shimmer, and harmonic-to-noise features.
        
        The motivation comes from prior research experience in Parkinson's-focused 
        cyber-physical systems and AI-based motor scoring.
        """
    )

    st.subheader("How the Pipeline Works")
    st.markdown(
        """
        1. Load biomedical voice features  
        2. Preprocess missing values  
        3. Train a machine learning classifier  
        4. Predict Parkinson's status  
        5. Display confidence and feature values  
        """
    )

with right_col:
    st.subheader("Model")
    st.write("Current model: **Random Forest Classifier**")
    st.write("Input type: **Tabular biomedical voice features**")
    st.write("Output: **Healthy vs Parkinson's Disease**")

st.divider()

st.header("Try the Classifier")

mode = st.radio(
    "Choose input mode:",
    ["Use example patient", "Enter values manually"],
    horizontal=True
)

example_values = {
    feature: 0.0 for feature in feature_columns
}

# A reasonable sample-like input can be edited by the user.
# These values are not diagnostic; they simply make the demo easier to use.
default_demo_values = {
    "MDVP:Fo(Hz)": 120.0,
    "MDVP:Fhi(Hz)": 150.0,
    "MDVP:Flo(Hz)": 80.0,
    "MDVP:Jitter(%)": 0.005,
    "MDVP:Jitter(Abs)": 0.00004,
    "MDVP:RAP": 0.003,
    "MDVP:PPQ": 0.003,
    "Jitter:DDP": 0.009,
    "MDVP:Shimmer": 0.03,
    "MDVP:Shimmer(dB)": 0.30
}

for feature in feature_columns:
    if feature in default_demo_values:
        example_values[feature] = default_demo_values[feature]

user_input = {}

if mode == "Use example patient":
    st.info("Using a pre-filled example. You can switch to manual mode to edit values.")
    user_input = example_values

    st.dataframe(pd.DataFrame([user_input]))

else:
    st.write("Enter biomedical voice measurements below.")

    cols = st.columns(3)

    for i, feature in enumerate(feature_columns):
        with cols[i % 3]:
            default_value = float(example_values.get(feature, 0.0))
            user_input[feature] = st.number_input(
                feature,
                value=default_value,
                format="%.6f"
            )

st.divider()

if st.button("Run Prediction", type="primary"):
    prediction, probability, input_df = predict(model, feature_columns, user_input)

    result_col, confidence_col = st.columns(2)

    with result_col:
        if prediction == 1:
            st.error("Prediction: Parkinson's Disease")
        else:
            st.success("Prediction: Healthy")

    with confidence_col:
        st.metric(
            "Estimated Probability of Parkinson's",
            f"{probability:.2%}"
        )

    st.subheader("Prediction Interpretation")

    if probability >= 0.75:
        st.write("The model is highly confident in a Parkinson's-positive prediction.")
    elif probability >= 0.50:
        st.write("The model leans toward Parkinson's-positive, but confidence is moderate.")
    elif probability >= 0.25:
        st.write("The model leans toward healthy, but confidence is moderate.")
    else:
        st.write("The model is highly confident in a healthy prediction.")

    st.subheader("Input Summary")
    st.dataframe(input_df)

st.divider()

st.subheader("About This Project")
st.write(
    """
    This prototype was built as a machine learning class project. 
    It uses the UCI Parkinson's voice dataset and demonstrates a complete applied ML pipeline:
    data preprocessing, model training, evaluation, saving model artifacts, and deploying an interactive demo.
    """
)