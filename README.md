# CS 4210 - Final Project | Hasti Abbasi | Spring 2026

## Parkinson's Disease Voice Classifier

This project is a machine learning application that predicts whether a voice sample is associated with Parkinson's Disease using biomedical voice measurements from the UCI Parkinson's dataset. The project includes a full machine learning pipeline, saved model artifacts, reusable inference code, and a Streamlit website.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Train the model

```bash
python train.py
```

This saves the trained model to:

```text
model/parkinsons_model.pkl
model/feature_columns.pkl
```

### 2. Run the website

```bash
streamlit run app.py
```

Open the link shown in the terminal:
```text
http://localhost:8501
```

## Main Files

- `train.py` → training pipeline  
- `predict.py` → inference logic  
- `app.py` → Streamlit demo  

## Dataset

Located at:

```text
data/parkinsons.data
```

Target:

```text
status (0 = Healthy, 1 = Parkinson's)
```
