import streamlit as st
import pandas as pd
import joblib
import numpy as np

# path to the trained model saved by train.py
MODEL_PATH = "model/parkinsons_model.pkl"
# path to teh saved feature col list
FEATURE_PATH = "model/feature_columns.pkl"
# dataset path used for loading example patients
DATA_PATH = "data/parkinsons.data"

# cache the model artifacts so streamlit doesn't reload on every interation
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    return model, feature_columns

# convert streamlit input values into the format expected by model
def make_prediction(model, feature_columns, input_dict):
    # convert input dictionary into a one row DF
    input_df = pd.DataFrame([input_dict])
    # match the exact feat order used during training
    input_df = input_df.reindex(columns=feature_columns)

    # predict class label + PD probability
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability, input_df

# groups features into categories to make input for easier
def group_features(feature_columns):
    # create empty feature groups for the frontend tabs
    groups = {
        "Frequency": [],
        "Jitter": [],
        "Shimmer": [],
        "Noise / Nonlinear": [],
        "Other": []
    }

    # assign each feature to a group based on keywords
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

# cache real example patients from dataset for inputs
@st.cache_data
def load_example_patients(feature_columns):
    # load original dataset so eamples come form  actual rows
    df = pd.read_csv(DATA_PATH)
    # remove hidden whitespace from col names
    df.columns = df.columns.str.strip()

    # select 1 real healthy sample and one real PD sample
    healthy_row = df[df["status"] == 0].iloc[0]
    pd_row = df[df["status"] == 1].iloc[0]

    # remove cols that aren't modle input faatures
    drop_cols = ["name", "sourcname", "status"]

    # convert rows into dictionaries matching the trained model's feature order
    healthy_row = healthy_row.drop(labels=[c for c in drop_cols if c in healthy_row.index])
    pd_row = pd_row.drop(labels=[c for c in drop_cols if c in pd_row.index])

    healthy_example = healthy_row.reindex(feature_columns).astype(float).to_dict()
    pd_example = pd_row.reindex(feature_columns).astype(float).to_dict()

    return healthy_example, pd_example

# genreate a simple explanation based on model confideence 
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

# configure browser tab title, icon, and page layout 
st.set_page_config(
    page_title="Parkinson's Voice Classifier",
    page_icon="🧠",
    layout="wide"
)

# load model, feature list, feature groups, and example patients
model, feature_columns = load_artifacts()
feature_groups = group_features(feature_columns)
healthy_example, pd_example = load_example_patients(feature_columns)

# sidebar nav (multi-page dashboard)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Single Prediction", "Batch Prediction", "Model Insights", "About"]
)

st.sidebar.divider()
st.sidebar.caption("Educational ML prototype")
st.sidebar.caption("Not for medical diagnosis")

# home page to explain project purpose and piepline 
if page == "Home":
    st.title("Parkinson's Disease Voice Classifier - Hasti Abbasi")
    st.subheader("An interactive ML prototype for Parkinson's prediction from biomedical voice features")

    # st.warning("This tool is for educational purposes only and is not a medical diagnosis system.")

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

# single pred page allows example or custom patient predition 
elif page == "Single Prediction":
    st.title("Single Patient Prediction")

    # lets user choose between real examples or manually entered vals
    input_mode = st.radio(
        "Choose input mode:",
        ["Parkinson's example", "Healthy example", "Custom manual input"],
        horizontal=True
    )

    # use real PD example from teh dataset
    if input_mode == "Parkinson's example":
        user_input = pd_example.copy()
        st.info("Using a pre-filled Parkinson's-like example from the dataset style.")
    # use real healthy example
    elif input_mode == "Healthy example":
        user_input = healthy_example.copy()
        st.info("Using a pre-filled healthy-like example from the dataset style.")
    # custom mode where build input wdiges are grouped by feature type
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

    # run trained model when user clicks the prediction button 
    if st.button("Run Prediction", type="primary"):
        prediction, probability, input_df = make_prediction(model, feature_columns, user_input)

        # show the prediction result + confidence score
        result_col, prob_col = st.columns([1, 1])

        with result_col:
            if prediction == 1:
                st.error("Prediction: Parkinson's Disease")
            else:
                st.success("Prediction: Healthy")

        with prob_col:
            # display probaility using text and progress bar
            st.metric("Probability of Parkinson's", f"{probability:.2%}")
            st.progress(float(probability))

        st.subheader("Explanation")
        st.write(explain_prediction(probability))

        st.subheader("Model Input Used")
        st.dataframe(input_df, use_container_width=True)

# batch prediction page allows csv upload + prediction for many samples
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
        # read teh uploaded csv into a DF
        batch_df = pd.read_csv(uploaded_file)

        # remove identifies cols if they appear in teh uploaded file
        if "name" in batch_df.columns:
            batch_df = batch_df.drop(columns=["name"])

        # remove true labels before prediction if teh uploaded CSv includes them
        if "status" in batch_df.columns:
            true_labels = batch_df["status"]
            batch_df = batch_df.drop(columns=["status"])
        else:
            true_labels = None

        # reorder uploaded data to match the model's expected feature order
        batch_df = batch_df.reindex(columns=feature_columns)

        # predict labels + PD probaiilities for every uploaded row
        predictions = model.predict(batch_df)
        probabilities = model.predict_proba(batch_df)[:, 1]

        # add model outputs to the uploaded data for display and download
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

# modle insights shows model summary and feature imporatnce
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

    # random forest exposes feature importance to exlain model
    if hasattr(model.named_steps["classifier"], "feature_importances_"):
        importances = model.named_steps["classifier"].feature_importances_

        # pair each feature name with importance score
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        st.subheader("Top Feature Importances")
        # show top features using table and bar chart
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
# about page shows project backgorund, dataset details, etc.
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

        """
    )