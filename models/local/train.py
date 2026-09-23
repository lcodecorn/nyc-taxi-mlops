"""
NYC Taxi Intelligence Engine — Training Pipeline
=================================================
Four-layer model stack:
    1. Demand        — LGBMRegressor   → trip_count
    2. Profitability — XGBRegressor    → target_rph
    3. Competition   — LGBMRegressor   → composite competition score
    4. Tip           — LogisticRegression → high_tip_zone (binary)

Fixes applied vs. previous version
-----------------------------------
- high_tip_zone computed BEFORE the train/valid/test split so every split
  carries the label without a SettingWithCopyWarning / silent NaN.
- All three split DataFrames are .copy() so downstream mutations are safe.
- competition_target is assigned directly onto each split frame (not sliced
  by position from a full-df Series), removing the index-alignment bug that
  could occur when df is re-sorted or filtered.
- MLflow run wraps ALL training (not just metrics) so params and artifacts
  are always logged even when a metric step raises.
- MLFLOW_TRACKING_URI default is http://mlflow:5000 (Docker service name);
  never use localhost inside a container.
- tip_valid_preds is computed and logged (was missing before).
- Artifact upload loop uses a context manager and logs the S3 key as a
  tag so every run is traceable back to its stored objects.
- Minor: one canonical split of X_* avoids re-slicing FEATURES repeatedly.
- Bayesian hyperparameter search (Optuna TPE) runs before the MLflow run.
  Best params per layer are logged as MLflow params alongside metrics.
  Set OPTUNA_N_TRIALS to control search budget (default 20).
- competition_target no longer sums lag_1h_trip_count + rolling_24h_rph_std +
  congestion_score — the first two are also in FEATURES, so the model was
  being handed two-thirds of its own label as input and just copying it
  back out, making competition_valid_rmse meaningless. Now built from
  net_pickup_flow/congestion_score (never raw model inputs, only their
  lag1h_ versions are), z-scored on train-only stats. See
  assign_competition_target.
"""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path

import boto3
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from optuna.samplers import TPESampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from monitoring.drift import save_reference

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

S3_BUCKET: str = os.environ["S3_BUCKET"]          # fail fast if missing
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", "http://mlflow:5000"    # never localhost in Docker
)
OPTUNA_N_TRIALS: int = int(os.getenv("OPTUNA_N_TRIALS", "20"))

INPUT_S3_KEY = "final/hourly_zone_timeseries.parquet"
MODEL_S3_PREFIX = "models/"
LOCAL_PARQUET = "hourly_zone_timeseries.parquet"

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "PUZone_enc", "PUBorough_enc",
    "hour", "day_of_week", "week_of_year", "month", "quarter", "day_of_year",
    "is_weekend", "is_rush", "is_night",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "lag1h_precipitation", "lag1h_temperature", "lag1h_wind_speed", "lag1h_frac_raining",
    "lag1h_frac_airport_pickup", "lag1h_frac_airport_dropoff",
    "rolling_airport_activity",
    "lag_1h_trip_count", "lag_24h_trip_count",
    "rolling_24h_trip_mean", "trip_count_delta_1h",
    "lag1h_demand_vs_baseline",
    "lag_1h_rph", "lag_2h_rph", "lag_24h_rph", "lag_168h_rph",
    "rolling_24h_rph_mean", "rolling_7d_rph_mean", "rolling_24h_rph_std",
    "rph_delta_1h", "rph_zscore", "rph_cv",
    "lag1h_dropoff_count", "lag1h_net_pickup_flow",
    "lag1h_avg_revenue", "lag1h_avg_tip", "lag1h_tip_rate",
    "lag1h_avg_distance_mi", "lag1h_avg_duration_min", "lag1h_avg_speed_mph",
    "lag1h_congestion_score",
    "relative_zone_activity",
]

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
# test gets the remainder implicitly

# Helpers
s3 = boto3.client("s3")


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def load_data() -> pd.DataFrame:
    print(f"Downloading s3://{S3_BUCKET}/{INPUT_S3_KEY} ...")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=INPUT_S3_KEY)
    with open(LOCAL_PARQUET, "wb") as fh:
        fh.write(obj["Body"].read())
    df = pd.read_parquet(LOCAL_PARQUET).sort_values("ts_hour").reset_index(drop=True)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    zone_enc = LabelEncoder()
    boro_enc = LabelEncoder()
    df = df.copy()
    df["PUZone_enc"] = zone_enc.fit_transform(df["PUZone"])
    df["PUBorough_enc"] = boro_enc.fit_transform(df["PUBorough"])
    return df, zone_enc, boro_enc


def add_tip_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill tip_rate NaNs. high_tip_zone is added per-split afterwards (see
    assign_tip_labels), since its threshold must come from train data only.
    """
    df = df.copy()
    df["tip_rate"] = df["tip_rate"].fillna(0)
    return df


def assign_tip_labels(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame
) -> float:
    """
    Compute the high-tip threshold from train_df only, then label all splits
    with it — avoids leaking valid/test tip_rate distribution into the label.
    """
    median_tip = train_df["tip_rate"].median()
    for split_df in (train_df, valid_df, test_df):
        split_df["high_tip_zone"] = (split_df["tip_rate"] > median_tip).astype(int)
    return median_tip


def split_data(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,   # full split frames
    pd.DataFrame, pd.DataFrame, pd.DataFrame,   # X splits
]:
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df  = df.iloc[valid_end:].copy()

    X_train = train_df[features]
    X_valid = valid_df[features]
    X_test  = test_df[features]

    print(
        f"Split → train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}"
    )
    return train_df, valid_df, test_df, X_train, X_valid, X_test


def assign_competition_target(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """
    Composite supply/demand-pressure proxy: oversupply (dropoffs outpacing
    pickups, i.e. -net_pickup_flow) plus congestion, both z-scored using
    train-only mean/std so valid/test don't leak into the label's scale —
    same pattern as the tip-rate threshold in assign_tip_labels.

    Built only from columns that are NOT in FEATURES (net_pickup_flow and
    congestion_score are same-hour values; the model only ever sees their
    lag1h_ versions as inputs). The previous version summed
    lag_1h_trip_count + rolling_24h_rph_std + congestion_score, but the
    first two are themselves entries in FEATURES — the model was handed
    2/3 of its own label as an input and just learned to copy it back out,
    making the validation RMSE meaningless as a signal of real skill.
    """
    oversupply_train = -train_df["net_pickup_flow"].fillna(0)
    congestion_train = train_df["congestion_score"].fillna(0)
    os_mean, os_std = oversupply_train.mean(), oversupply_train.std()
    cg_mean, cg_std = congestion_train.mean(), congestion_train.std()

    for split_df in (train_df, valid_df, test_df):
        oversupply = (-split_df["net_pickup_flow"].fillna(0) - os_mean) / (os_std + 1e-9)
        congestion = (split_df["congestion_score"].fillna(0) - cg_mean) / (cg_std + 1e-9)
        split_df["competition_target"] = oversupply + congestion


# Bayesian hyperparameter search (Optuna TPE)
def tune_lgbm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_valid: pd.DataFrame, y_valid: pd.Series,
    n_trials: int,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 800),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 150),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        return _mae(y_valid, model.predict(X_valid))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_xgb(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_valid: pd.DataFrame, y_valid: pd.Series,
    n_trials: int,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 800),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        return _mae(y_valid, model.predict(X_valid))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_logistic(
    X_train_arr: np.ndarray, y_train: pd.Series,
    X_valid_arr: np.ndarray, y_valid: pd.Series,
    n_trials: int,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_train_arr, y_train)
        return 1.0 - float(accuracy_score(y_valid, model.predict(X_valid_arr)))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def upload_and_log_artifacts(run: mlflow.ActiveRun) -> None:
    for artifact in sorted(MODEL_DIR.iterdir()):
        if not artifact.is_file():
            continue
        mlflow.log_artifact(str(artifact))
        s3_key = MODEL_S3_PREFIX + artifact.name
        with open(artifact, "rb") as fh:
            s3.upload_fileobj(fh, S3_BUCKET, s3_key)
        mlflow.set_tag(f"s3_{artifact.stem}", f"s3://{S3_BUCKET}/{s3_key}")
        print(f"  {artifact.name} → s3://{S3_BUCKET}/{s3_key}")


# Main
def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("taxi_model_training")

    # 1. Data
    df = load_data()
    df, zone_encoder, borough_encoder = encode_categoricals(df)
    df = add_tip_label(df)

    # Keep only features that actually exist in this dataset
    active_features = [c for c in FEATURES if c in df.columns]
    print(f"Active features: {len(active_features)} / {len(FEATURES)}")

    train_df, valid_df, test_df, X_train, X_valid, X_test = split_data(df, active_features)

    # high_tip_zone threshold computed from train split only, then applied to all splits
    assign_tip_labels(train_df, valid_df, test_df)

    # Competition target (z-score stats from train split only, applied to all splits)
    assign_competition_target(train_df, valid_df, test_df)
    comp_target_train = train_df["competition_target"]
    comp_target_valid = valid_df["competition_target"]

    tip_imputer = SimpleImputer(strategy="median")
    X_train_tip = tip_imputer.fit_transform(X_train)
    X_valid_tip = tip_imputer.transform(X_valid)

    # ── 2. Bayesian hyperparameter search ────────────────────────────────────
    print(f"\nRunning Bayesian search ({OPTUNA_N_TRIALS} trials per model)…")

    print("  [Layer 1] Tuning demand model (LGBM)…")
    demand_params = {
        **tune_lgbm(X_train, train_df["trip_count"], X_valid, valid_df["trip_count"], OPTUNA_N_TRIALS),
        "random_state": 42,
    }

    print("  [Layer 2] Tuning profitability model (XGB)…")
    profit_params = {
        **tune_xgb(X_train, train_df["target_rph"], X_valid, valid_df["target_rph"], OPTUNA_N_TRIALS),
        "objective": "reg:squarederror",
        "random_state": 42,
    }

    print("  [Layer 3] Tuning competition model (LGBM)…")
    comp_params = {
        **tune_lgbm(X_train, comp_target_train, X_valid, comp_target_valid, OPTUNA_N_TRIALS),
        "random_state": 42,
    }

    print("  [Layer 4] Tuning tip model (LogisticRegression)…")
    tip_params = {
        **tune_logistic(X_train_tip, train_df["high_tip_zone"], X_valid_tip, valid_df["high_tip_zone"], OPTUNA_N_TRIALS),
        "max_iter": 1000,
        "random_state": 42,
    }

    # ── 3. MLflow run wraps ALL training ─────────────────────────────────────
    with mlflow.start_run(run_name="taxi_model_training") as run:
        mlflow.set_tag("run_type", "training")
        mlflow.log_params({
            "train_size":      len(train_df),
            "valid_size":      len(valid_df),
            "test_size":       len(test_df),
            "n_features":      len(active_features),
            "model_dir":       str(MODEL_DIR),
            "train_ratio":     TRAIN_RATIO,
            "valid_ratio":     VALID_RATIO,
            "optuna_n_trials": OPTUNA_N_TRIALS,
        })
        mlflow.log_params({f"demand_{k}": v  for k, v in demand_params.items() if k != "random_state"})
        mlflow.log_params({f"profit_{k}": v  for k, v in profit_params.items() if k not in ("random_state", "objective")})
        mlflow.log_params({f"comp_{k}": v    for k, v in comp_params.items()   if k != "random_state"})
        mlflow.log_params({f"tip_{k}": v     for k, v in tip_params.items()    if k not in ("random_state", "max_iter")})

        # ── Layer 1 — Demand ─────────────────────────────────────────────────
        print("\n[Layer 1] Training demand model …")
        demand_model = LGBMRegressor(**demand_params)
        demand_model.fit(X_train, train_df["trip_count"])

        d_train_pred = demand_model.predict(X_train)
        d_valid_pred = demand_model.predict(X_valid)
        mlflow.log_metrics({
            "demand_train_mae": _mae(train_df["trip_count"], d_train_pred),
            "demand_valid_mae": _mae(valid_df["trip_count"], d_valid_pred),
        })
        joblib.dump(demand_model, MODEL_DIR / "demand_model.pkl")

        # ── Layer 2 — Profitability ──────────────────────────────────────────
        print("[Layer 2] Training profitability model …")
        profit_model = XGBRegressor(**profit_params)
        profit_model.fit(X_train, train_df["target_rph"])

        p_train_pred = profit_model.predict(X_train)
        p_valid_pred = profit_model.predict(X_valid)
        mlflow.log_metrics({
            "profit_train_rmse": _rmse(train_df["target_rph"], p_train_pred),
            "profit_valid_rmse": _rmse(valid_df["target_rph"], p_valid_pred),
        })
        joblib.dump(profit_model, MODEL_DIR / "profitability_model.pkl")

        # ── Layer 3 — Competition ────────────────────────────────────────────
        print("[Layer 3] Training competition model …")
        competition_model = LGBMRegressor(**comp_params)
        competition_model.fit(X_train, comp_target_train)

        c_train_pred = competition_model.predict(X_train)
        c_valid_pred = competition_model.predict(X_valid)
        mlflow.log_metrics({
            "competition_train_rmse": _rmse(comp_target_train, c_train_pred),
            "competition_valid_rmse": _rmse(comp_target_valid, c_valid_pred),
        })
        joblib.dump(competition_model, MODEL_DIR / "competition_model.pkl")

        # ── Layer 4 — Tip classifier ─────────────────────────────────────────
        print("[Layer 4] Training tip model …")
        tip_model = LogisticRegression(**tip_params)
        tip_model.fit(X_train_tip, train_df["high_tip_zone"])

        t_train_pred = tip_model.predict(X_train_tip)
        t_valid_pred = tip_model.predict(X_valid_tip)
        mlflow.log_metrics({
            "tip_train_accuracy": accuracy_score(train_df["high_tip_zone"], t_train_pred),
            "tip_valid_accuracy": accuracy_score(valid_df["high_tip_zone"], t_valid_pred),
        })
        joblib.dump(tip_model,    MODEL_DIR / "tip_model.pkl")
        joblib.dump(tip_imputer,  MODEL_DIR / "tip_imputer.pkl")

        # ── Drift reference (read by monitoring/drift.py) ────────────────────
        save_reference(
            train_df, active_features, demand_model,
            valid_metrics={
                "demand_valid_mae": _mae(valid_df["trip_count"], d_valid_pred),
                "profit_valid_rmse": _rmse(valid_df["target_rph"], p_valid_pred),
            },
            out_dir=MODEL_DIR,
        )

        # ── Encoders + metadata ──────────────────────────────────────────────
        joblib.dump(zone_encoder,    MODEL_DIR / "zone_encoder.pkl")
        joblib.dump(borough_encoder, MODEL_DIR / "borough_encoder.pkl")
        joblib.dump(active_features, MODEL_DIR / "feature_list.pkl")
        joblib.dump(
            {"n_features": len(active_features), "train_size": len(train_df)},
            MODEL_DIR / "metadata.pkl",
        )

        # ── Upload all artifacts ─────────────────────────────────────────────
        print("\nUploading artifacts …")
        upload_and_log_artifacts(run)

    print("\n  TRAINING COMPLETE")


if __name__ == "__main__":
    main()
