from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Task functions
def start():
    print("Pipeline started")

def process_data():
    print("Processing data...")

def end():
    print("Pipeline finished")

# DAG definition
with DAG(
    dag_id="basic_pipeline",
    start_date=datetime(2026, 5, 4),
    schedule=None,  # manual trigger
    catchup=False
) as dag:

    task_start = PythonOperator(
        task_id="start_task",
        python_callable=start
    )

    task_process = PythonOperator(
        task_id="process_task",
        python_callable=process_data
    )

    task_end = PythonOperator(
        task_id="end_task",
        python_callable=end
    )

    # Task flow
    task_start >> task_process >> task_end