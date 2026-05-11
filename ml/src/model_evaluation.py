"""
Model Evaluation Module

This module:
1. Loads best tuned model
2. Evaluates model on test dataset
3. Saves final evaluation metrics

Author: Atul
"""

import json
import math

import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model():
    """
    Evaluate best tuned machine learning model.
    """

    # Load transformed test features
    X_test = joblib.load(
        "/opt/airflow/data/features/X_test.pkl"
    )

    # Load test labels
    y_test = pd.read_csv(
        "/opt/airflow/data/splits/y_test.csv"
    )

    print("Test datasets loaded")

    # Load best tuned model
    model = joblib.load(
        "/opt/airflow/ml/artifacts/best_model.pkl"
    )

    print("Best model loaded")

    # Predict
    predictions = model.predict(X_test)

    print("Predictions generated")

    # Calculate metrics
    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = math.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions,
    )

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2_score": float(r2),
    }

    print("Final evaluation metrics calculated")
    print(metrics)

    # Save metrics
    with open(
        "/opt/airflow/ml/artifacts/final_metrics.json",
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("Final metrics saved successfully")