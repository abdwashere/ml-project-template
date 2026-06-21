"""
This is the file YOU customize for your specific model.

Everything else in this template (API, monitoring, Docker, CI/CD)
calls these functions and doesn't need to know what's inside them.

To use this template for a new project:
1. Replace the model creation logic in build_model()
2. Replace the training loop in train()
3. Replace the prediction logic in predict()
4. Update params.yaml with your hyperparameters

As long as these three functions keep their input/output shapes,
nothing else in the template needs to change.
"""

import pickle
import os
from sklearn.ensemble import RandomForestClassifier  # swap for your model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def build_model(hyperparameters: dict):
    """
    Create an untrained model instance.
    Replace this with your actual model (XGBoost, PyTorch nn.Module, etc).
    """
    return RandomForestClassifier(
        n_estimators=hyperparameters.get("n_estimators", 100),
        max_depth=hyperparameters.get("max_depth", 10),
        random_state=42,
    )


def load_data(train_path: str, test_size: float, random_state: int):
    """
    Replace this with your actual data loading logic.
    Must return: X_train, X_test, y_train, y_test
    """
    import pandas as pd

    df = pd.read_csv(train_path)
    X = df.drop(columns=["label"])  # adjust to your target column name
    y = df["label"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train(model, X_train, y_train):
    """
    Train the model. Replace with your training loop if not using
    a scikit-learn-style .fit() API (e.g. PyTorch training loop).
    """
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """
    Return a dict of metrics. These get logged to MLflow automatically.
    Add/remove metrics relevant to your problem (e.g. AUC-PR for
    imbalanced classification, RMSE for regression).
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model, features: dict):
    """
    This is what the API calls for live predictions.
    `features` is a dict from the API request — convert it to
    whatever shape your model expects.
    """
    import pandas as pd

    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]

    # If your model supports probabilities, include them — useful
    # for confidence-based monitoring (see monitoring/monitor.py)
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(X)[0]))

    return {
        "prediction": prediction.item() if hasattr(prediction, "item") else prediction,
        "confidence": confidence,
    }