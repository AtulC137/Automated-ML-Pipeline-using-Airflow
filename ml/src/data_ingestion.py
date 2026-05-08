import pandas as pd
from pathlib import Path


def load_data(file_path: str):

    file_extension = Path(file_path).suffix

    if file_extension == ".csv":
        data = pd.read_csv(file_path)

    elif file_extension == ".json":
        data = pd.read_json(file_path)

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    print(f"Dataset loaded successfully: {file_path}")
    print(f"Shape: {data.shape}")

    return data