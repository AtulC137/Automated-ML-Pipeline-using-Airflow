"""
Basic Airflow ML Pipeline

This DAG performs:
1. Data ingestion
2. Data preprocessing

Author: Atul
"""

from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from ml.src.data_ingestion import load_data
from ml.src.data_preprocessing import preprocess_data
from ml.src.data_validation import validate_data
from ml.src.data_splitting import split_data
from ml.src.feature_engineering import engineer_features
from ml.src.hyperparameter_tuning import tune_models
from ml.src.model_evaluation import evaluate_model

def ingest_data():
    """
    Load raw housing dataset.
    """

    load_data("/opt/airflow/data/housing.csv")

def validate_dataset():
    """
    Validate input dataset.
    """

    validate_data("/opt/airflow/data/housing.csv")

def preprocess_dataset():
    """
    Preprocess dataset and save cleaned dataset.
    """

    preprocess_data(
        "/opt/airflow/data/housing.csv"
    )

def split_dataset():
    """
    Split cleaned dataset into train and test sets.
    """

    split_data(
        input_path="/opt/airflow/data/processed/cleaned_data.csv",
        target_column="median_house_value",
    )

def feature_engineering_dataset():
    """
    Perform feature engineering on split datasets.
    """

    engineer_features()


def hyperparameter_tuning_dataset():
    """
    Tune machine learning models.
    """

    tune_models()

def model_evaluation_dataset():
    """
    Evaluate best tuned machine learning model.
    """

    evaluate_model()

with DAG(
    dag_id="basic_pipeline",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
    tags=["ml_pipeline"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )
    
    validate_task = PythonOperator(
    task_id="validate_data",
    python_callable=validate_dataset,  
    )
    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_dataset,
    )
    split_task = PythonOperator(
    task_id="split_data",
    python_callable=split_dataset,
    )
    feature_engineering_task = PythonOperator(
    task_id="feature_engineering",
    python_callable=feature_engineering_dataset,
    )
    hyperparameter_tuning_task = PythonOperator(
    task_id="hyperparameter_tuning",
    python_callable=hyperparameter_tuning_dataset,
    )
    model_evaluation_task = PythonOperator(
    task_id="model_evaluation",
    python_callable=model_evaluation_dataset,
    )


    # Task dependency
    (
    ingest_task
    >> validate_task
    >> preprocess_task
    >> split_task
    >> feature_engineering_task
    >> hyperparameter_tuning_task
    >> model_evaluation_task
    )