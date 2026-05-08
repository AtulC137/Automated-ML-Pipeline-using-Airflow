"""
Generic Data Splitting Module

This module:
1. Loads cleaned dataset
2. Splits dataset into train and test sets
3. Saves split datasets

Author: Atul
"""

import os

from sklearn.model_selection import train_test_split

from ml.src.data_ingestion import load_data


def split_data(
    input_path,
    target_column,
    test_size=0.2,
    random_state=42,
):
    """
    Split dataset into train and test sets.

    Parameters
    ----------
    input_path : str
        Path to cleaned dataset.

    target_column : str
        Target column name.

    test_size : float, optional
        Test split ratio.

    random_state : int, optional
        Random seed for reproducibility.

    Raises
    ------
    ValueError
        If target column does not exist.
    """

    # Load cleaned dataset
    df = load_data(input_path)

    print("Cleaned dataset loaded successfully")

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found"
        )

    # Features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Create split folder
    os.makedirs(
        "/opt/airflow/data/splits",
        exist_ok=True,
    )

    # Save split datasets
    X_train.to_csv(
        "/opt/airflow/data/splits/X_train.csv",
        index=False,
    )

    X_test.to_csv(
        "/opt/airflow/data/splits/X_test.csv",
        index=False,
    )

    y_train.to_csv(
        "/opt/airflow/data/splits/y_train.csv",
        index=False,
    )

    y_test.to_csv(
        "/opt/airflow/data/splits/y_test.csv",
        index=False,
    )

    print("Train-test split completed successfully")