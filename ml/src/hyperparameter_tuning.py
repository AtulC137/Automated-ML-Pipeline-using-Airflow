"""
Hyperparameter Tuning Module

This module:
1. Loads feature-engineered datasets
2. Tunes multiple machine learning models
3. Selects best model
4. Saves best model and metrics

Author: Atul
"""

import json
import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


def tune_models():
    """
    Tune multiple machine learning models.
    """

    # Load transformed datasets
    X_train = joblib.load(
        "/opt/airflow/data/features/X_train.pkl"
    )

    X_test = joblib.load(
        "/opt/airflow/data/features/X_test.pkl"
    )

    y_train = pd.read_csv(
        "/opt/airflow/data/splits/y_train.csv"
    )

    y_test = pd.read_csv(
        "/opt/airflow/data/splits/y_test.csv"
    )

    print("Datasets loaded successfully")

    # Models and parameter grids
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {},
        },

        "DecisionTree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "max_depth": [5, 10, 20],
                "min_samples_split": [2, 5],
            },
        },

        "RandomForest": {
            "model": RandomForestRegressor(),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [10, 20],
            },
        },
    }

    best_model = None
    best_score = -1
    best_model_name = ""

    model_scores = {}

    # Tune all models
    for model_name, config in models.items():

        print(f"Tuning {model_name}")

        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=3,
            scoring="r2",
            n_jobs=-1,
        )

        # Train + tune
        grid_search.fit(
            X_train,
            y_train.values.ravel(),
        )

        # Best tuned model
        tuned_model = grid_search.best_estimator_

        # Predict
        predictions = tuned_model.predict(X_test)

        # Evaluate
        score = r2_score(
            y_test,
            predictions,
        )

        model_scores[model_name] = {
            "r2_score": float(score),
            "best_params": grid_search.best_params_,
        }

        print(f"{model_name} Score: {score}")
        print(f"Best Params: {grid_search.best_params_}")

        # Track best model
        if score > best_score:
            best_score = score
            best_model = tuned_model
            best_model_name = model_name

    print(f"Best Model: {best_model_name}")
    print(f"Best Score: {best_score}")

    # Create artifacts folder
    os.makedirs(
        "/opt/airflow/ml/artifacts",
        exist_ok=True,
    )

    # Save best model
    joblib.dump(
        best_model,
        "/opt/airflow/ml/artifacts/best_model.pkl",
    )

    print("Best model saved successfully")

    # Save scores
    with open(
        "/opt/airflow/ml/artifacts/model_scores.json",
        "w",
    ) as file:

        json.dump(
            model_scores,
            file,
            indent=4,
        )

    print("Model scores saved successfully")