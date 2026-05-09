import streamlit as st
import pandas as pd
import joblib
import numpy as np


MODEL_PATH = "model/parkinsons_model.pkl"
FEATURE_PATH = "model/feature_columns.pkl"
DATA_PATH = "data/parkinsons.data"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    return model, feature_columns


def make_prediction(model, feature_columns, input_dict):
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability, input_df


def group_features(feature_columns):
    groups = {
        "Frequency": [],
        "Jitter": [],
        "Shimmer": [],
        "Noise / Nonlinear": [],
        "Other": []
    }

    for feature in feature_columns:
        lower = feature.lower()

        if "fo" in lower or "fhi" in lower or "flo" in lower:
            groups["Frequency"].append(feature)
        elif "jitter" in lower or "rap" in lower or "ppq" in lower or "ddp" in lower:
            groups["Jitter"].append(feature)
        elif "shimmer" in lower or "apq" in lower or "dda" in lower:
            groups["Shimmer"].append(feature)
        elif "nhr" in lower or "hnr" in lower or "rpde" in lower or "dfa" in lower or "ppe" in lower or "spread" in lower or "d2" in lower:
            groups["Noise / Nonlinear"].append(feature)
        else:
            groups["Other"].append(feature)

    return groups


@st.cache_data
def load_example_patients(feature_columns):
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Real examples from dataset
    healthy_row = df[df["status"] == 0].iloc[0]
    pd_row = df[df["status"] == 1].iloc[0]

    # Remove non-feature columns
    drop_cols = ["name", "sourcname", "status"]

    healthy_row = healthy_row.drop(labels=[c for c in drop_cols if c in healthy_row.index])
    pd_row = pd_row.drop(labels=[c for c in drop_cols if c in pd_row.index])

    healthy_example = healthy_row.reindex(feature_columns).astype(float).to_dict()
    pd_example = pd_row.reindex(feature_columns).astype(float).to_dict()

    return healthy_example, pd_example


def explain_prediction(probability):
    if probability >= 0.75:
        return (
            "The model is highly confident that this sample resembles the Parkinson's class. "
            "This may be associated with stronger irregularities in voice stability, such as jitter, shimmer, "
            "or nonlinear acoustic patterns."
        )
    elif probability >= 0.50:
        return (
            "The model leans toward Parkinson's Disease, but the confidence is moderate. "
            "A few acoustic features may be pushing the prediction upward."
        )
    elif probability >= 0.25:
        return (
            "The model leans toward the healthy class, but the confidence is moderate. "
            "Some features may still overlap with Parkinson's-like voice patterns."
        )
    else:
        return (
            "The model is highly confident that this sample resembles the healthy class. "
            "The input voice features appear less consistent with Parkinson's-like acoustic instability."
        )


st.set_page_config(
    page_title="Parkinson's Voice Classifier",
    page_icon="🧠",
    layout="wide"
)

model, feature_columns = load_artifacts()
feature_groups = group_features(feature_columns)
healthy_example, pd_example = load_example_patients(feature_columns)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Single Prediction", "Batch Prediction", "Model Insights", "About"]
)

st.sidebar.divider()
st.sidebar.caption("Educational ML prototype")
st.sidebar.caption("Not for medical diagnosis")

if page == "Home":
    st.title("Parkinson's Disease Voice Classifier")
    st.subheader("An interactive ML prototype for Parkinson's prediction from biomedical voice features")

    st.warning("This tool is for educational purposes only and is not a medical diagnosis system.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Input Modality", "Voice Features")
    col2.metric("Model Type", "Random Forest")
    col3.metric("Prediction Task", "Binary Classification")

    st.divider()

    st.markdown(
        """
        ### What this app does

        This demo predicts whether a voice sample is more consistent with a healthy control or a Parkinson's Disease sample.
        It uses biomedical voice measurements such as:

        - Fundamental frequency
        - Jitter
        - Shimmer
        - Noise-to-harmonics ratio
        - Nonlinear vocal features

        ### Why this matters

        Parkinson's Disease can affect speech and vocal stability. Acoustic features provide a low-cost way to explore
        machine learning-based screening tools.
        """
    )

    st.markdown(
        """
        ### Project pipeline

        `UCI voice dataset → preprocessing → Random Forest training → saved model → Streamlit demo`
        """
    )

elif page == "Single Prediction":
    st.title("Single Patient Prediction")

    input_mode = st.radio(
        "Choose input mode:",
        ["Parkinson's example", "Healthy example", "Custom manual input"],
        horizontal=True
    )

    if input_mode == "Parkinson's example":
        user_input = pd_example.copy()
        st.info("Using a pre-filled Parkinson's-like example from the dataset style.")
    elif input_mode == "Healthy example":
        user_input = healthy_example.copy()
        st.info("Using a pre-filled healthy-like example from the dataset style.")
    else:
        user_input = {}

        tabs = st.tabs(list(feature_groups.keys()))

        for tab, group_name in zip(tabs, feature_groups.keys()):
            with tab:
                group_features_list = feature_groups[group_name]

                if not group_features_list:
                    st.write("No features in this group.")

                cols = st.columns(2)

                for i, feature in enumerate(group_features_list):
                    default_value = float(pd_example.get(feature, 0.0))
                    with cols[i % 2]:
                        user_input[feature] = st.number_input(
                            feature,
                            value=default_value,
                            format="%.6f"
                        )

    if input_mode != "Custom manual input":
        st.subheader("Input Feature Values")
        st.dataframe(pd.DataFrame([user_input]), use_container_width=True)

    st.divider()

    if st.button("Run Prediction", type="primary"):
        prediction, probability, input_df = make_prediction(model, feature_columns, user_input)

        result_col, prob_col = st.columns([1, 1])

        with result_col:
            if prediction == 1:
                st.error("Prediction: Parkinson's Disease")
            else:
                st.success("Prediction: Healthy")

        with prob_col:
            st.metric("Probability of Parkinson's", f"{probability:.2%}")
            st.progress(float(probability))

        st.subheader("Explanation")
        st.write(explain_prediction(probability))

        st.subheader("Model Input Used")
        st.dataframe(input_df, use_container_width=True)

elif page == "Batch Prediction":
    st.title("Batch Prediction")

    st.write(
        """
        Upload a CSV file with the same feature columns used during training.
        The app will return predictions and Parkinson's probabilities for every row.
        """
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        if "name" in batch_df.columns:
            batch_df = batch_df.drop(columns=["name"])

        if "status" in batch_df.columns:
            true_labels = batch_df["status"]
            batch_df = batch_df.drop(columns=["status"])
        else:
            true_labels = None

        batch_df = batch_df.reindex(columns=feature_columns)

        predictions = model.predict(batch_df)
        probabilities = model.predict_proba(batch_df)[:, 1]

        results = batch_df.copy()
        results["prediction"] = predictions
        results["prediction_label"] = np.where(predictions == 1, "Parkinson's Disease", "Healthy")
        results["parkinsons_probability"] = probabilities

        st.subheader("Batch Results")
        st.dataframe(results, use_container_width=True)

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions as CSV",
            csv,
            "parkinsons_predictions.csv",
            "text/csv"
        )

elif page == "Model Insights":
    st.title("Model Insights")

    st.subheader("Model Summary")
    st.write("The deployed model is a Random Forest classifier trained on UCI Parkinson's voice features.")

    st.markdown(
        """
        ### Evaluation metrics used during training

        - Accuracy
        - Precision
        - Recall
        - F1-score
        - ROC-AUC

        These metrics are used because the task is a binary classification problem.
        """
    )

    if hasattr(model.named_steps["classifier"], "feature_importances_"):
        importances = model.named_steps["classifier"].feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        st.subheader("Top Feature Importances")
        st.dataframe(importance_df, use_container_width=True)

        st.bar_chart(importance_df.set_index("Feature").head(15))

    st.subheader("Interpretability Note")
    st.write(
        """
        Feature importance shows which variables the Random Forest used most often to split the data.
        This does not prove clinical causality, but it helps explain which acoustic measurements were most useful
        for the classifier.
        """
    )

elif page == "About":
    st.title("About This Project")

    st.markdown(
        """
        ### Project idea

        This project builds a Parkinson's Disease classifier using biomedical voice features.

        ### Motivation

        The project is motivated by prior research experience in Parkinson's-focused healthcare AI,
        including work on multimodal motor scoring and scalable AI systems for clinical assessment.

        ### Dataset

        The project uses the UCI Parkinson's dataset. The target variable is `status`,
        where `1` indicates Parkinson's Disease and `0` indicates healthy.

        ### Main system components

        - `train.py`: training and evaluation pipeline
        - `predict.py`: reusable prediction logic
        - `app.py`: Streamlit frontend
        - `model/`: saved trained model and feature list

        ### Disclaimer

        This prototype is for class demonstration and educational use only.
        It is not intended for diagnosis or medical decision-making.
        """
    )