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


DATA_PATH = "data/parkinsons.data"
MODEL_DIR = "model"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)

    if "name" in df.columns:
        df = df.drop(columns=["name"])

    X = df.drop(columns=["status"])
    y = df["status"]

    return X, y


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


def tune_random_forest(X_train, y_train):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5, 10]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    return metrics, classification_report(y_test, predictions)


def save_artifacts(model, feature_columns):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, "parkinsons_model.pkl"))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Training Random Forest model with hyperparameter tuning...")
    best_model, best_params, best_cv_score = tune_random_forest(X_train, y_train)

    print("\nBest parameters:")
    print(best_params)

    print("\nBest cross-validation F1:")
    print(best_cv_score)

    metrics, report = evaluate_model(best_model, X_test, y_test)

    print("\nTest metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification report:")
    print(report)

    save_artifacts(best_model, list(X.columns))

    print("\nSaved model artifacts to model/.")


if __name__ == "__main__":
    main()