"""
Taxi model training DAG — EC2 self-stop variant.

Runs only when manually triggered (or chained from the ingestion DAG).

Pipeline:
    check_retrain -> [train_model] -> forecast -> stop_instance

train_model is skipped when the ingestion DAG finds no drift (conf retrain=false).
"""

from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

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

IMDS_BASE = "http://169.254.169.254/latest"


def _imds_token() -> str:
    resp = requests.put(
        f"{IMDS_BASE}/api/token",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        timeout=5,
    )
    resp.raise_for_status()
    return resp.text


def _current_instance_id(token: str) -> str:
    resp = requests.get(
        f"{IMDS_BASE}/meta-data/instance-id",
        headers={"X-aws-ec2-metadata-token": token},
        timeout=5,
    )
    resp.raise_for_status()
    return resp.text


def stop_self(**context):
    """Stop the EC2 instance this Airflow worker is running on."""
    import boto3

    token = _imds_token()
    instance_id = _current_instance_id(token)

    print(f"Pipeline finished — stopping EC2 instance {instance_id}")
    ec2 = boto3.client("ec2")
    ec2.stop_instances(InstanceIds=[instance_id])


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
    dag_id="taxi_model_training_ec2_self_stop",
    description="Train models, generate forecast, then stop the EC2 host",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["taxi", "training", "ec2"],
) as dag:

    check_retrain = BranchPythonOperator(
        task_id="check_retrain",
        python_callable=choose_path,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"{BASH_PREFIX} python models/ec2/train.py",
        execution_timeout=timedelta(hours=4),
    )

    forecast = BashOperator(
        task_id="forecast",
        trigger_rule="none_failed_min_one_success",
        bash_command=f"{BASH_PREFIX} python models/ec2/forecast.py",
        execution_timeout=timedelta(hours=1),
    )

    stop_instance = PythonOperator(
        task_id="stop_instance",
        python_callable=stop_self,
        trigger_rule="all_success",
    )

    check_retrain >> [train_model, forecast]
    train_model >> forecast >> stop_instance
