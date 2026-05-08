from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from ml.src.data_ingestion import load_data

# from ml.data_ingestion import load_data

def ingest_data():
    load_data("/opt/airflow/data/housing.csv")


with DAG(
    dag_id="basic_pipeline",
    start_date=datetime(2026, 5, 8),
    schedule=None,
    catchup=False,
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    ingest_task