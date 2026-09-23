"""
Taxi model training DAG.

Runs when manually triggered or when the ingestion DAG detects drift.

Dependencies: taxi_data_pipeline (provides merged and engineered features)

Pipeline:
    check_retrain -> [train_model] -> forecast

train_model is skipped when triggered with conf {"retrain": false}.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

AIRFLOW_ROOT = "/opt/airflow"

BASH_PREFIX = (
    f"cd {AIRFLOW_ROOT} && "
    "export PYTHONPATH=/opt/airflow:${PYTHONPATH:-} && "
)


def choose_path(**context):
    """
    Skip train_model when triggered with conf {"retrain": false} (no drift
    detected upstream) and go straight to forecasting with the current models.
    Manual triggers without conf retrain as before.
    """
    conf = context["dag_run"].conf or {}
    retrain = str(conf.get("retrain", True)).strip().lower() not in ("false", "0", "no")
    print(f"retrain={retrain} (conf={conf})")
    return "train_model" if retrain else "forecast"


with DAG(
    dag_id="taxi_model_training",
    description="Merge, feature engineering, training and forecasting",
    schedule=None,  # manual or triggered by taxi_data_ingestion_local
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["taxi", "training"],
) as dag:

    check_retrain = BranchPythonOperator(
        task_id="check_retrain",
        python_callable=choose_path,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"{BASH_PREFIX} python models/local/train.py",
    )

    forecast = BashOperator(
        task_id="forecast",
        trigger_rule="none_failed_min_one_success",
        bash_command=f"{BASH_PREFIX} python models/local/forecast.py",
    )

    check_retrain >> [train_model, forecast]
    train_model >> forecast