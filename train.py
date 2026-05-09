import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# path to the local dataset used for training
DATA_PATH = "data/parkinsons.data"
# folder where trained model artifcats will be saved
MODEL_DIR = "model"

# loads dataset, removed non-feature cols, and returns X/y for training
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # remove extra spaces from col names 
    df.columns = df.columns.str.strip()

    # status is target label
    if "status" not in df.columns:
        raise ValueError("Expected target column 'status' was not found.")

    # separate the prediction target before cleaning feature cols
    y = df["status"]

    # remove target col so X only contains input feats
    X = df.drop(columns=["status"])

    # drop known identifier / bad columns
    columns_to_drop = ["name", "sourcname"]
    X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])

    # keep only numeric features
    X = X.apply(pd.to_numeric, errors="coerce")

    # drop columns that became entirely NaN
    X = X.dropna(axis=1, how="all")

    print("Final training features:")
    print(X.columns.tolist())
    print("Number of features:", X.shape[1])

    return X, y


# function to build log regression, svm, and random forest models
def build_models():
    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42))
        ]),

        "svm": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(probability=True, random_state=42))
        ]),

        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(random_state=42))
        ])
    }

    return models

# trains + tunes a random forest classifier using cross-validation
def tune_random_forest(X_train, y_train):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # hyperparameters tested during GridSearchCV
    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5, 10]
    }

    # use StratifiedKFold to keep healthy/PD ratio similar in each fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # train multiple versions of the model + select teh best one by F1 score
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    # fit all candidate models on training data
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# evaluate the trained model on unseen test data
def evaluate_model(model, X_test, y_test):
    # predicted class labels: health = 0 and PD = 1
    predictions = model.predict(X_test)
    # prob of class 1 (PD)
    probabilities = model.predict_proba(X_test)[:, 1]

    # store recall, f1, precision, accuracy, and roc_auc for metrics
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    return metrics, classification_report(y_test, predictions)

# save trained model + feature order 
def save_artifacts(model, feature_columns):
    # make model directory if doesn't exist alr
    os.makedirs(MODEL_DIR, exist_ok=True)

    # save full sklearn pipeline
    joblib.dump(model, os.path.join(MODEL_DIR, "parkinsons_model.pkl"))
    # save feature names so the app can provide inputs in the same order
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))

# run full training workflow
def main():
    # load claned features + target labels
    X, y = load_data()

    # print("Current working directory:", os.getcwd())
    # print("Training data shape:", X.shape)
    # print("Features used for training:")
    # print(X.columns.tolist())
    # print("Contains name?", "name" in X.columns)

    # split data into training + testing sets 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Training Random Forest model with hyperparameter tuning...")
    # train + tune final model
    best_model, best_params, best_cv_score = tune_random_forest(X_train, y_train)

    print("\nBest parameters:")
    print(best_params)

    print("\nBest cross-validation F1:")
    print(best_cv_score)

    # evaluate the selected model 
    metrics, report = evaluate_model(best_model, X_test, y_test)

    print("\nTest metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification report:")
    print(report)

    # save model so the website can run preds without retraining
    save_artifacts(best_model, list(X.columns))

    print("\nSaved model artifacts to model/.")


if __name__ == "__main__":
    main()