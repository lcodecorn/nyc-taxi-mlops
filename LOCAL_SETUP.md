# Local Development Setup

Run the full NYC Taxi MLOps stack on your machine using `docker-compose.yml`.

## Architecture

```
Airflow (webserver + scheduler + worker + triggerer)
  → taxi_data_ingestion_local DAG
      → download taxi & weather data → S3
      → merge + feature engineering
      → drift check (monitoring/drift.py) → S3 + MLflow
  → taxi_model_training DAG (triggered when drift is detected, or manually)
      → train → MLflow → forecast → S3

MLflow server  →  http://localhost:5000
Airflow UI     →  http://localhost:8080
Forecast API   →  http://localhost:7860
```

---

## Prerequisites

- Docker Desktop (WSL2 backend recommended on Windows) — at least **4 GB RAM** and **10 GB disk** allocated
- AWS account with an S3 bucket and an IAM user that has read/write access to it
- PowerShell (Windows) or any shell

---

## Step 1 — Create your .env file

Copy the example and fill in your values:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set:

```env
COMPOSE_PROJECT_NAME=nyc-taxi-mlops

# AWS credentials — required locally (no IAM role on your machine)
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
AWS_REGION=eu-north-1
S3_BUCKET=taxi-mlops-nyc

# Airflow — use your local UID to avoid permission issues on Linux/macOS
# On Windows leave as 1000
AIRFLOW_UID=1000
AIRFLOW_PROJ_DIR=.

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Email alerts (optional — leave the defaults to skip)
ALERT_EMAIL=you@gmail.com
SMTP_USER=you@gmail.com
SMTP_PASSWORD=xxxx-xxxx-xxxx-xxxx

# Optuna
OPTUNA_N_TRIALS=20
```

> On Linux/macOS run `echo "AIRFLOW_UID=$(id -u)" >> .env` to set the correct UID.

---

## Step 2 — Build images and initialise Airflow

```powershell
docker compose up airflow-init
```

Wait for the line `exited with code 0`. This creates the Airflow database and the default admin user (`airflow` / `airflow`).

---

## Step 3 — Start the full stack

```powershell
docker compose up -d
```

Check that all services are healthy:

```powershell
docker compose ps
```

All containers should show `healthy` or `running` within ~2 minutes.

---

## Step 4 — Open the UIs

| Service | URL | Credentials |
|---|---|---|
| Airflow | http://localhost:8080 | `airflow` / `airflow` |
| MLflow | http://localhost:5000 | — |
| Forecast API | http://localhost:7860 | — |

---

## Step 5 — Run the pipeline

1. Open the Airflow UI at http://localhost:8080
2. Unpause **`taxi_data_ingestion_local`** and **`taxi_model_training`** (a triggered run of a paused DAG stays queued)
3. Click **Trigger DAG** on the ingestion DAG → optionally pass `{"month": "2024-01"}` to target a specific month; leave empty to use the previous calendar month. Add `"force_retrain": true` to retrain regardless of drift.
4. After ingestion, `detect_drift` compares the new month to the training data. If drift is detected, `taxi_model_training` is triggered automatically; otherwise it is skipped. The first run always retrains (no reference exists yet).
5. Monitor MLflow at http://localhost:5000 — `taxi_model_training` for training runs, `taxi_drift_monitoring` for monthly drift reports

> The ingestion DAG has `catchup=True`, so unpausing it backfills up to 12 months — each month can trigger a training run if drift is found.

---

## Stopping and cleaning up

Stop without removing data:
```powershell
docker compose down
```

Stop and remove all volumes (wipes the Airflow DB and MLflow DB — S3 data is unaffected):
```powershell
docker compose down -v
```

---

## Troubleshooting

**Containers exit immediately on first `up -d`**
- Run `airflow-init` first (Step 2) — the webserver and scheduler depend on it completing successfully.

**`AIRFLOW_UID` warning on Linux**
- Set `AIRFLOW_UID=$(id -u)` in `.env` so container files are owned by your user, not root.

**MLflow can't write artifacts**
- Confirm `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set in `.env` — unlike EC2, your local machine has no IAM role.
- Check the bucket name and region match `S3_BUCKET` and `AWS_REGION`.

**Port conflicts**
- Airflow uses `8080`, MLflow uses `5000`, the API uses `7860`. Stop any local processes on those ports before starting the stack.

**Low memory warning during init**
- The warning is non-fatal if init exits with code 0. Increase Docker Desktop memory allocation if the stack becomes unstable.

**Celery Flower (optional monitoring UI)**
```powershell
docker compose --profile flower up -d
```
Opens at http://localhost:5555.
