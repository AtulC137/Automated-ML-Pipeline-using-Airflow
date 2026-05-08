"""
Feature Engineering Module

This module:
1. Loads train-test datasets
2. Encodes categorical features
3. Scales numerical features
4. Saves transformed datasets
5. Saves preprocessing pipeline

Author: Atul
"""

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


def engineer_features():
    """
    Perform feature engineering on train and test datasets.
    """

    # Load datasets
    X_train = pd.read_csv(
        "/opt/airflow/data/splits/X_train.csv"
    )

    X_test = pd.read_csv(
        "/opt/airflow/data/splits/X_test.csv"
    )

    print("Train and test datasets loaded")

    # Separate column types
    categorical_columns = (
        X_train.select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    numerical_columns = (
        X_train.select_dtypes(
            exclude=["object"]
        )
        .columns
        .tolist()
    )

    print(f"Categorical columns: {categorical_columns}")
    print(f"Numerical columns: {numerical_columns}")

    # Numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            )
        ]
    )

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    # Fit on training data
    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    # Transform test data
    X_test_transformed = preprocessor.transform(
        X_test
    )

    print("Feature engineering completed")

    # Create output folders
    os.makedirs(
        "/opt/airflow/data/features",
        exist_ok=True,
    )

    os.makedirs(
        "/opt/airflow/ml/artifacts",
        exist_ok=True,
    )

    # Save transformed datasets
    joblib.dump(
        X_train_transformed,
        "/opt/airflow/data/features/X_train.pkl",
    )

    joblib.dump(
        X_test_transformed,
        "/opt/airflow/data/features/X_test.pkl",
    )

    # Save preprocessor
    joblib.dump(
        preprocessor,
        "/opt/airflow/ml/artifacts/preprocessor.pkl",
    )

    print("Feature engineered datasets saved")
    print("Preprocessor saved successfully")