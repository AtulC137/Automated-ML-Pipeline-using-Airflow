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


def ingest_data():
    """
    Load raw housing dataset.
    """

    load_data("/opt/airflow/data/housing.csv")


def preprocess_dataset():
    """
    Preprocess housing dataset and save processed files.
    """

    preprocess_data("/opt/airflow/data/housing.csv")


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

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_dataset,
    )

    # Task dependency
    ingest_task >> preprocess_task