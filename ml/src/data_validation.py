"""
Generic Data Validation Module

This module performs generic dataset validation.

Validation checks:
1. File exists
2. Dataset is not empty
3. Dataset has columns

Author: Atul
"""

import os
import pandas as pd

from ml.src.data_ingestion import load_data


def validate_data(file_path):
    """
    Validate input dataset.

    Parameters
    ----------
    file_path : str
        Path to dataset.

    Raises
    ------
    FileNotFoundError
        If dataset file does not exist.

    ValueError
        If dataset is empty or invalid.
    """

    # Check file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Load dataset
    df = load_data(file_path)

    print("Dataset loaded for validation")

    # Check empty dataframe
    if df.empty:
        raise ValueError("Dataset is empty")

    print("Dataset is not empty")

    # Check columns exist
    if len(df.columns) == 0:
        raise ValueError(
            "Dataset has no columns"
        )

    print(f"Dataset contains {len(df.columns)} columns")

    print("Data validation completed successfully")