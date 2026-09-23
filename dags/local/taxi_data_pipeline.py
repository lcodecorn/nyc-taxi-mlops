"""
Monthly NYC taxi ingestion pipeline (S3-only).

Responsibilities:
- Determine target month
- Download taxi trip data
- Download weather data
- Prune old S3 files

- Validate, merge and engineer features
- Check for data drift and trigger training when it is detected

This DAG does NOT:
- train models
- generate forecasts
"""

import sys
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator



sys.path.insert(0, "/opt/airflow")

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


def get_target_month(**context):
    def _normalize_month(value: str | None) -> str | None:
        if not value:
            return None
        value = value.strip()
        if value.upper() == "YYYY-MM":
            return None
        try:
            datetime.strptime(value, "%Y-%m")
            return value
        except ValueError:
            raise ValueError(
                f"Invalid month format: {value!r}. Expected YYYY-MM or leave blank to use the previous month."
            )

    dag_run = context.get("dag_run")

    # Trigger DAG with config {"month": "2024-01"}
    if dag_run and dag_run.conf and dag_run.conf.get("month"):
        month = _normalize_month(dag_run.conf["month"])
        if month:
            print(f"Month from trigger conf: {month}")
            return month

    # DAG params override
    params = context.get("params") or {}
    month_override = _normalize_month(params.get("month"))

    if month_override:
        print(f"Month from DAG param: {month_override}")
        return month_override

    # Scheduled or manual run without explicit month
    interval_start = context.get("data_interval_start") or context.get("execution_date")
    if not interval_start:
        interval_start = datetime.utcnow()

    prev_month = interval_start.replace(day=1) - timedelta(days=1)
    month = prev_month.strftime("%Y-%m")

    print(f"Month from schedule/execution date: {month}")
    return month


def run_download(**context):
    from ingestion.download import download_month_to_s3

    month = context["ti"].xcom_pull(task_ids="get_month")

    if not month:
        raise ValueError("No month received from get_month")

    s3_key = download_month_to_s3(month)

    context["ti"].xcom_push(
        key="s3_key",
        value=s3_key,
    )


def run_fetch_weather(**context):
    from ingestion.weather import sync_weather_month_to_s3

    month = context["ti"].xcom_pull(task_ids="get_month")

    if not month:
        raise ValueError("No month received from get_month")

    s3_key = sync_weather_month_to_s3(
        month,
        skip_if_exists=True,
    )

    context["ti"].xcom_push(
        key="weather_s3_key",
        value=s3_key,
    )


def run_prune(**context):
    from ingestion.download import prune_s3_months

    month = context["ti"].xcom_pull(task_ids="get_month")

    if not month:
        raise ValueError("No month received from get_month")

    deleted = prune_s3_months(
        latest_month=month,
        keep_months=12,
    )

    context["ti"].xcom_push(
        key="deleted_keys",
        value=deleted,
    )


def run_pytest_validation(**context):
    """Run pytest to validate downloaded data."""
    import subprocess

    month = context["ti"].xcom_pull(task_ids="get_month")

    if not month:
        raise ValueError("No month received from get_month")

    # Run pytest on the validation script
    # Use -rxs to show skip reasons without failing on skipped tests
    pythonpath = f"{AIRFLOW_ROOT}:{os.environ.get('PYTHONPATH', '')}"
    
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            f"{AIRFLOW_ROOT}/tests/test_data_validation_local.py",
            "-v",
            "--tb=short",
            "-rxs",  # Show extra summary info for skipped tests
        ],
        env={
            **os.environ,
            "TEST_MONTH": month,
            "PYTHONPATH": pythonpath,
        },
        capture_output=True,
        text=True,
    )

    print("PYTEST OUTPUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    # Exit codes: 0=passed, 1=failed, 5=no tests collected/all skipped
    # Only fail on actual test failures (return code 1)
    if result.returncode == 1:
        raise RuntimeError("Data validation tests failed")
    elif result.returncode not in (0, 5):
        raise RuntimeError(f"Pytest exited with unexpected code {result.returncode}")

    context["ti"].xcom_push(
        key="validation_passed",
        value=True,
    )


def run_drift_check(**context):
    """
    Compare the freshly engineered month against the training reference.
    Returns True when the models should be retrained (see monitoring/drift.py);
    trigger with conf {"force_retrain": true} to retrain regardless.
    """
    from monitoring.drift import run_drift_check as _run_drift_check

    month = context["ti"].xcom_pull(task_ids="get_month")
    if not month:
        raise ValueError("No month received from get_month")

    report = _run_drift_check(month)

    dag_run = context.get("dag_run")
    force = bool(dag_run and dag_run.conf and dag_run.conf.get("force_retrain"))
    if force:
        print("force_retrain set in trigger conf — retraining regardless of drift")

    return report["should_retrain"] or force


with DAG(
    dag_id="taxi_data_ingestion_local",
    description="Monthly taxi ingestion pipeline (local docker)",
    schedule="@monthly",
    start_date=days_ago(365),
    catchup=True,
    max_active_runs=1,
    default_args=default_args,
    tags=["taxi", "ingestion"],
) as dag:

    get_month = PythonOperator(
        task_id="get_month",
        python_callable=get_target_month,
    )

    prune_old_data = PythonOperator(
        task_id="prune_old_data",
        python_callable=run_prune,
    )

    download_data = PythonOperator(
        task_id="download_data",
        python_callable=run_download,
    )

    fetch_weather = PythonOperator(
        task_id="fetch_weather",
        python_callable=run_fetch_weather,
    )

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=run_pytest_validation,
    )
    
    merge_everything = BashOperator(
        task_id="merge_everything",
        bash_command=f"{BASH_PREFIX} python preprocess/merge.py",
    )

    features2 = BashOperator(
        task_id="features2",
        bash_command=f"{BASH_PREFIX} python preprocess/features.py",
        env={
            **os.environ,
            "TARGET_MONTH": "{{ ti.xcom_pull(task_ids='get_month') }}",
        },
    )

    # Skips trigger_training when no drift is detected
    detect_drift = ShortCircuitOperator(
        task_id="detect_drift",
        python_callable=run_drift_check,
    )

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="taxi_model_training",
        conf={"retrain": True},
        wait_for_completion=False,
    )

    get_month >> prune_old_data >> [download_data, fetch_weather] >> validate_data >> merge_everything >> features2 >> detect_drift >> trigger_training