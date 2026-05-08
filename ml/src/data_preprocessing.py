"""
Generic Data Preprocessing Module

This module:
1. Loads dataset
2. Removes missing values
3. Saves cleaned dataset

Supports:
- CSV
- JSON

Author: Atul
"""

import os

from ml.src.data_ingestion import load_data


def preprocess_data(input_path):
    """
    Preprocess dataset.

    Parameters
    ----------
    input_path : str
        Path to input dataset.
    """

    # Load dataset
    df = load_data(input_path)

    print("Dataset loaded successfully")

    # Remove missing values
    df = df.dropna()

    print("Missing values removed")
    print(f"New shape: {df.shape}")

    # Create processed folder
    os.makedirs(
        "/opt/airflow/data/processed",
        exist_ok=True,
    )

    # Save cleaned dataset
    df.to_csv(
        "/opt/airflow/data/processed/cleaned_data.csv",
        index=False,
    )

    print("Cleaned dataset saved successfully")