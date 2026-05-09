# CS 4210 - Final Project | Hasti Abbasi | Spring 2026

## Parkinson's Disease Voice Classifier

This project is a machine learning application that predicts whether a voice sample is associated with Parkinson's Disease using biomedical voice measurements from the UCI Parkinson's dataset. The project includes a full machine learning pipeline, saved model artifacts, reusable inference code, and a Streamlit website.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

### Option 1 — Run full pipeline 

#### 1. Train the model

```bash
python train.py
```

This generates:

```text
model/parkinsons_model.pkl
model/feature_columns.pkl
```

#### 2. Run the website

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

### Option 2 — Run website only (no training)

Because the `model/` folder already contains:

```text
parkinsons_model.pkl
feature_columns.pkl
```

you can skip training and directly run:

```bash
streamlit run app.py
```

---

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
