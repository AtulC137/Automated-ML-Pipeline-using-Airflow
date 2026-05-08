"""
Generic Data Ingestion Module

Supports:
1. CSV files
2. JSON files

Author: Atul
"""

import os
import pandas as pd


def load_data(file_path):
    """
    Load dataset from CSV or JSON file.

    Parameters
    ----------
    file_path : str
        Path to dataset file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If file does not exist.

    ValueError
        If unsupported file format is used.
    """

    # Check file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    # CSV support
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    # JSON support
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)

    else:
        raise ValueError(
            "Unsupported file format. Use CSV or JSON."
        )

    print("Dataset loaded successfully")
    print(f"Shape: {df.shape}")

    return df