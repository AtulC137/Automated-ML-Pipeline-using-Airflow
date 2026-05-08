'''
Docstring for mlops-project.ml.src.data_preprocessing
raw CSV
→ clean data
→ split train/test
→ save processed datasets
'''


import os
import pandas as pd

from sklearn.model_selection import train_test_split


def preprocess_data(input_path):

    # Load dataset
    df = pd.read_csv(input_path)

    print("Dataset loaded successfully")
    print(df.shape)

    # Remove missing values
    df = df.dropna()

    print("Missing values removed")
    print(df.shape)

    # Features and target
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create processed folder if not exists
    os.makedirs("/opt/airflow/data/processed", exist_ok=True)

    # Save processed files
    X_train.to_csv("/opt/airflow/data/processed/X_train.csv", index=False)
    X_test.to_csv("/opt/airflow/data/processed/X_test.csv", index=False)

    y_train.to_csv("/opt/airflow/data/processed/y_train.csv", index=False)
    y_test.to_csv("/opt/airflow/data/processed/y_test.csv", index=False)

    print("Processed files saved successfully")