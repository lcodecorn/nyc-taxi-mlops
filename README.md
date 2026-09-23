# NYC Taxi Ride Demand & Opportunity MLOps

An end-to-end MLOps pipeline that ingests NYC yellow taxi trip + weather data,
engineers an hourly zone-level feature store, trains a 4-layer model stack,
generates ranked "where should a driver go next" forecasts, and serves them
through a FastAPI service and a Streamlit map dashboard.

The pipeline runs on Airflow and can be deployed in two ways:

- **Local** — Docker Compose stack on your machine (`docker-compose.yml`)
- **EC2** — A scheduled, self-stopping EC2 box driven by Lambda + EventBridge
  (`docker-compose.ec2.yml`)

---

## Architecture

```
                         ┌───────────────────────────┐
                         │   Ingestion DAG (Airflow) │
                         └───────────────────────────┘
                                      │
            get_month → prune_old_data → [download_data, fetch_weather]
                                      │
                                validate_data
                                      │
                         merge_everything (preprocess/merge.py)
                                      │
                              features (preprocess/features.py)
                                      │
                                      ▼
                         S3: final/hourly_zone_timeseries.parquet
                                      │
                       detect_drift (monitoring/drift.py) ──► S3: monitoring/drift_<month>.json
                                      │                        MLflow: taxi_drift_monitoring
        ┌─────────────────────────────────────────────────────────┐
        │ local: triggered only when drift is detected            │
        │ ec2:   always triggered, conf retrain=<drift detected>  │
        └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                         ┌───────────────────────────┐
                         │   Training DAG (Airflow)  │
                         └───────────────────────────┘
          check_retrain → [train_model] → forecast (→ stop_instance on EC2)
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                  ▼
           MLflow (params, metrics,            S3: models/*.pkl + drift reference
           model artifacts)                    S3: forecasts/*.csv
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                  ▼
              FastAPI Forecast API             Streamlit Map App
              (api/serve.py, port 7860)        (app/app.py)
```

---

## Model stack

Four models trained per zone-hour and combined into a single
`opportunity_score` (`preprocess/features.py` builds the features,
`models/*/train.py` trains, `models/*/forecast.py` / `api/engine.py` predict):

| Layer | Model | Target |
|---|---|---|
| Demand | LGBMRegressor | `trip_count` |
| Profitability | XGBRegressor | `target_rph` (revenue/hour) |
| Competition | LGBMRegressor | composite competition score |
| Tip | LogisticRegression | `high_tip_zone` (binary) |

Hyperparameters for all four layers are tuned with Optuna (TPE) before
the final MLflow-tracked training run (`OPTUNA_N_TRIALS`, default 20).

---

## Repository layout

```
dags/
  local/
    taxi_data_pipeline.py     # taxi_data_ingestion_local DAG (triggers training on drift)
    taxi_model_pipeline.py    # taxi_model_training DAG (manual or drift-triggered)
  ec2/
    ingestion.py              # taxi_data_ingestion_ec2 DAG (always triggers training DAG)
    training.py               # taxi_model_training_ec2_self_stop DAG (retrains only on drift)

ingestion/
  download.py                 # download_month_to_s3 / prune_s3_months
  weather.py                  # sync_weather_month_to_s3 (Open-Meteo)

preprocess/
  merge.py                     # merges raw trips + weather + zones -> S3 enriched parquet
  features.py                  # DuckDB feature engineering -> hourly zone timeseries

models/
  local/
    train.py                   # Optuna-tuned training (logs to MLflow)
    forecast.py                # generates forecast CSV, writes to S3
  ec2/
    train.py                   # Optuna-tuned training + email alert (logs to MLflow)
    forecast.py                # forecast generation (EC2 variant)

api/
  serve.py                      # FastAPI app (/health, /meta, /forecast)
  engine.py                      # cached model + feature-store loading, inference, ranking

app/
  app.py                        # Streamlit zone-opportunity map (reads forecasts from S3)
  dockerfile                     # Streamlit container (Hugging Face Spaces)

mlflow_space/
  Dockerfile                     # MLflow tracking server as a private Hugging Face Space (used by EC2)

lambda/
  start_ec2.py                   # EventBridge -> start EC2 (monthly)
  stop_ec2.py                    # EventBridge -> safety-net stop EC2

docker/
  Dockerfile.airflow             # Airflow image (local + EC2)
  Dockerfile.airflow-ec2          # EC2 variant (same image; see comments for boot requirements)
  Dockerfile.api                  # FastAPI image

monitoring/
  drift.py                      # monthly feature / target / performance drift check + retrain decision

tests/
  test_data_validation_local.py  # validates landed S3 data (skips without AWS config)
  test_data_validation_ec2.py
  test_drift.py                  # unit tests for drift statistics + retrain decision

requirements/
  base.txt / airflow.txt / api.txt

.github/workflows/ci.yml          # lint, tests, Docker builds
conftest.py                       # puts repo root on sys.path for pytest
docker-compose.yml                # local dev stack
docker-compose.ec2.yml             # EC2 stack (IAM role creds, localhost-only ports)
```

---

## Pipelines

### Ingestion DAG (`taxi_data_ingestion_local` / `taxi_data_ingestion_ec2`)

Runs `@monthly`. Resolves the target month (from trigger config, DAG param, or
the previous calendar month), then:

1. `prune_old_data` — drops S3 data older than 12 months
2. `download_data` / `fetch_weather` — fetch trip parquet + Open-Meteo weather, in parallel
3. `validate_data` — runs `tests/test_data_validation_*.py` against the new data
4. `merge_everything` — `preprocess/merge.py`
5. `features` — `preprocess/features.py`
6. `detect_drift` — `monitoring/drift.py` compares the new month to the training reference (see [Drift monitoring](#drift-monitoring))
7. `trigger_training` — kicks off the training DAG
   - local: only when drift is detected (`ShortCircuitOperator`)
   - EC2: always, with conf `{"retrain": <drift detected>}` so the training DAG still forecasts and stops the instance

Trigger with conf `{"force_retrain": true}` to retrain regardless of drift.

> Airflow creates DAGs paused (`DAGS_ARE_PAUSED_AT_CREATION`). Unpause **both** the
> ingestion and training DAGs, otherwise triggered training runs stay queued.

### Training DAG (`taxi_model_training` / `taxi_model_training_ec2_self_stop`)

Manual trigger, or triggered by the ingestion DAG:

1. `check_retrain` — branches on conf `retrain` (default `true`); `false` skips `train_model`
2. `train_model` — trains the 4-layer model stack, logs params/metrics/models to MLflow,
   uploads models to S3, and writes the drift reference (`drift_reference.parquet`, `drift_baseline.json`)
3. `forecast` — loads production models, generates ranked forecasts, writes to S3
4. *(EC2 only)* `stop_instance` — stops the EC2 host via its IAM role

---

## Drift monitoring

`monitoring/drift.py` runs monthly after feature engineering and compares the
new month of `final/hourly_zone_timeseries.parquet` against a sample of the
rows the production models were trained on (`models/drift_reference.parquet`).

| Signal | Method | Retrain when (env override) |
|---|---|---|
| Feature drift | PSI per feature, weighted by demand-model importance; KS logged as supporting evidence | weighted share of features with PSI > 0.2 is > 0.3 (`DRIFT_PSI_THRESHOLD`, `DRIFT_SHARE_THRESHOLD`) |
| Target drift | PSI on `trip_count`, `target_rph` | PSI > 0.2 (`DRIFT_TARGET_PSI_THRESHOLD`) |
| Performance drift | past forecasts in `forecasts/` joined to the actuals that just landed | realised MAE/RMSE > 1.2× validation (`DRIFT_PERF_TOLERANCE`) |

- Calendar features (`month`, `quarter`, `week_of_year`, `day_of_year`) and label-encoded zones are excluded.
- The same calendar month of the reference is used when it has ≥ 10k rows (handles weather seasonality); otherwise the full reference.
- With no reference yet (first run), it always requests a retrain.

Results go to `s3://$S3_BUCKET/monitoring/drift_<month>.json` / `_features.csv`
and the MLflow experiment `taxi_drift_monitoring` (one run per month, `psi_<feature>`
metrics for comparing months). Run standalone with
`TARGET_MONTH=2024-06 python monitoring/drift.py`.

---

## Forecast API

```
uvicorn api.serve:app --reload
```

| Endpoint | Description |
|---|---|
| `GET /health` | liveness check |
| `GET /meta` | latest data hour + valid `rank_by` options |
| `GET /forecast?hours=24&top=10&rank_by=opportunity` | ranked zone-hour forecast |

`rank_by` options: `demand`, `profit`, `competition`, `tip`, `opportunity`.

Models and the feature store are loaded from S3 once per process and cached
in memory (`api/engine.py`).

---

## Streamlit app

Demo: https://lcodecorn-nyc-taxi.hf.space

`app/app.py` reads the latest `forecasts/*.csv` from S3, joins it
with the NYC taxi zone GeoJSON, and renders a choropleth map with per-zone
demand/profitability/competition/opportunity scores, plus a sortable zone
ranking table. Deployed as a Docker-based Hugging Face Space (`app/dockerfile`,
port 7860).

---

## Setup

- **Local development** (Docker Compose, Airflow + MLflow + API on your
  machine): see [LOCAL_SETUP.md](LOCAL_SETUP.md)
- **EC2 deployment** (scheduled, self-stopping EC2 box with Lambda +
  EventBridge automation): see [EC2_SETUP.md](EC2_SETUP.md)

Both setups read configuration from a `.env` file — copy `.env.example` to
get started.

After pulling changes to `requirements/airflow.txt`, rebuild the Airflow image
(`docker compose build`, or `docker compose -f docker-compose.ec2.yml build`).

## Tests & CI

```
pytest tests/ -v
```

`.github/workflows/ci.yml` runs on pushes and PRs to `main`: ruff (syntax
errors / undefined names only), the test suite (data validation tests skip
without AWS config), and Docker builds of the API, Airflow and Streamlit images.

## Services & ports (local)

| Service | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| Forecast API | http://localhost:7860 |
