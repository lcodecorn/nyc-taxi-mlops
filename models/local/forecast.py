# NYC TAXI INTELLIGENCE ENGINE — FORECAST PIPELINE
# Loads the 4 trained models and generates ranked zone-hour opportunities
# for the next N hours.
#
# Usage:
#   python forecast.py                          # next 24 hours, all zones
#   python forecast.py --hours 48              # next 48 hours
#   python forecast.py --hours 12 --top 20    # top 20 zones only
#   python forecast.py --start "2024-06-01 08:00"  # custom start time

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
import joblib
import boto3
import io
import sys
from dotenv import load_dotenv
import os
import mlflow
load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("taxi_forecasting")

warnings.filterwarnings("ignore")

# CONFIG
MODEL_S3_PREFIX = "models/"
DATA_S3_KEY = "final/hourly_zone_timeseries.parquet"
TMP_TIMESERIES = "/tmp/hourly_zone_timeseries.parquet"
OUTPUT_S3_PREFIX = "forecasts/"
OUTPUT_DIR = Path("forecasts")
OUTPUT_DIR.mkdir(exist_ok=True)
s3 = boto3.client("s3")


def download_timeseries_from_s3() -> None:
    """Download hourly feature store from S3 to a local temp path."""
    obj = s3.get_object(Bucket=S3_BUCKET, Key=DATA_S3_KEY)
    with open(TMP_TIMESERIES, "wb") as f:
        f.write(obj["Body"].read())


# Weights for final opportunity score (must sum to 1.0)
SCORE_WEIGHTS = {
    "demand":      0.35,
    "profit":      0.35,
    "competition": 0.20,
    "tip":         0.10,
}

# Same-hour aggregates that training only sees as lag1h_* (see preprocess/features.py).
# At forecast time these are reconstructed by shifting the raw historical
# columns forward by 1h, same as lag_1h_rph / lag_1h_trip_count.
RAW_LAG1H_COLS = [
    "avg_revenue", "avg_tip", "tip_rate",
    "avg_distance_mi", "avg_duration_min", "avg_speed_mph",
    "congestion_score", "dropoff_count", "net_pickup_flow",
    "frac_airport_pickup", "frac_airport_dropoff",
    "precipitation", "temperature", "wind_speed", "frac_raining",
]

# Longest look-back any feature needs (lag_168h_rph, rolling_7d_rph_mean).
MAX_LOOKBACK_HOURS = 168

# LOAD MODELS
def load_models():
    print("Loading models from S3...")
    def load_s3_joblib(key):
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_S3_PREFIX + key)
        return joblib.load(io.BytesIO(obj["Body"].read()))
    models = {
        "demand":      load_s3_joblib("demand_model.pkl"),
        "profit":      load_s3_joblib("profitability_model.pkl"),
        "competition": load_s3_joblib("competition_model.pkl"),
        "tip":         load_s3_joblib("tip_model.pkl"),
        "tip_imputer": load_s3_joblib("tip_imputer.pkl"),
        "zone_enc":    load_s3_joblib("zone_encoder.pkl"),
        "borough_enc": load_s3_joblib("borough_encoder.pkl"),
    }
    features = load_s3_joblib("feature_list.pkl")
    metadata = load_s3_joblib("metadata.pkl")
    print(f"  {len(features)} features | trained on {metadata['train_size']:,} rows")
    return models, features


# BUILD HISTORICAL CONTEXT
def load_history(zone_enc, borough_enc):
    print("Loading historical context from S3...")
    download_timeseries_from_s3()
    df = pd.read_parquet(TMP_TIMESERIES).sort_values("ts_hour")

    # The store also holds zone-hours below the training trip_count threshold,
    # so it can contain zones the encoders never saw — drop them (same as
    # api/engine.py). Features are per zone, so other zones are unaffected.
    known_mask = (
        df["PUZone"].notna() & df["PUZone"].isin(zone_enc.classes_)
        & df["PUBorough"].notna() & df["PUBorough"].isin(borough_enc.classes_)
    )
    dropped = (~known_mask).sum()
    if dropped:
        print(f"  Dropping {dropped:,} history rows with zones unseen by the trained encoders")
    df = df[known_mask]

    df["PUZone_enc"] = zone_enc.transform(df["PUZone"])
    df["PUBorough_enc"] = borough_enc.transform(df["PUBorough"])
    return df


def resolve_forecast_start(df_hist: pd.DataFrame, start_override: str | None) -> datetime:
    """
    Pick forecast start from the feature store unless --start is given.

    Default: first hour AFTER max(ts_hour) in s3://.../hourly_zone_timeseries.parquet
    """
    min_ts = pd.Timestamp(df_hist["ts_hour"].min())
    max_ts = pd.Timestamp(df_hist["ts_hour"].max())

    print(f"  Feature store range : {min_ts} → {max_ts}  ({len(df_hist):,} rows)")

    if start_override:
        start_dt = datetime.strptime(start_override, "%Y-%m-%d %H:%M")
        print(f"  Forecast start      : {start_dt}  (manual --start)")
        return start_dt

    start_dt = (max_ts + pd.Timedelta(hours=1)).to_pydatetime()
    print(f"  Dataset last hour   : {max_ts}")
    print(f"  Forecast start      : {start_dt}  (auto: last hour + 1)")
    return start_dt


def assert_forecast_window(future: pd.DataFrame, start_dt: datetime, n_hours: int) -> None:
    """Fail fast if generated hours do not match the resolved start."""
    expected_first = pd.Timestamp(start_dt)
    actual_first = pd.Timestamp(future["ts_hour"].min())
    actual_last = pd.Timestamp(future["ts_hour"].max())
    expected_last = expected_first + pd.Timedelta(hours=n_hours - 1)

    if actual_first != expected_first:
        raise ValueError(
            f"Forecast window mismatch: expected first hour {expected_first}, got {actual_first}"
        )
    if actual_last != expected_last:
        raise ValueError(
            f"Forecast window mismatch: expected last hour {expected_last}, got {actual_last}"
        )
    print(
        f"  Forecast window OK  : {actual_first} → {actual_last}  ({n_hours} hour(s))"
    )


# GENERATE FUTURE ROWS
def build_future_frame(df_hist, start_dt, n_hours, zone_enc):
    future_hours = [start_dt + timedelta(hours=h) for h in range(n_hours)]
    zones = df_hist[
        ["PUZone", "PUBorough", "PUZone_enc", "PUBorough_enc"]
    ].drop_duplicates()

    rows = []
    for ts in future_hours:
        tmp = zones.copy()
        tmp["ts_hour"] = ts
        rows.append(tmp)

    future = pd.concat(rows, ignore_index=True)

    # Temporal features
    future["hour"]         = future["ts_hour"].dt.hour
    future["day_of_week"]  = future["ts_hour"].dt.dayofweek
    future["week_of_year"] = future["ts_hour"].dt.isocalendar().week.astype(int)
    future["month"]        = future["ts_hour"].dt.month
    future["quarter"]      = future["ts_hour"].dt.quarter
    future["day_of_year"]  = future["ts_hour"].dt.dayofyear

    future["is_weekend"] = (future["day_of_week"] >= 5).astype(int)
    future["is_rush"]    = future["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    future["is_night"]   = (
        future["hour"].isin(range(22, 24)) | future["hour"].isin(range(0, 6))
    ).astype(int)

    # Cyclical encodings
    future["hour_sin"] = np.sin(2 * np.pi * future["hour"] / 24)
    future["hour_cos"] = np.cos(2 * np.pi * future["hour"] / 24)
    future["dow_sin"]  = np.sin(2 * np.pi * future["day_of_week"] / 7)
    future["dow_cos"]  = np.cos(2 * np.pi * future["day_of_week"] / 7)

    return future


# ATTACH LAG & ROLLING FEATURES
def history_before(df_hist, start_dt):
    """
    History strictly before the forecast start. Nothing at or after it may
    feed a forecast — matters for backtests whose start lies inside the store.
    """
    hist = df_hist[df_hist["ts_hour"] < pd.Timestamp(start_dt)]
    if hist.empty:
        raise ValueError(f"No history before forecast start {start_dt}")
    return hist


def attach_lag_features(future, df_hist):
    """
    Rebuild the lag / rolling features exactly as preprocess/features.py does:
    on a gap-free zone × hour grid, so "24h" means 24 hours rather than the
    last 24 rows a zone happened to have trips in.

    Pass the FULL store as df_hist: it is cut at the forecast start here, but
    its last hour says how far the data actually extends. Grid hours before
    the start and within the store are observed: no stored row means no
    pickups, so trip_count = 0 and every per-trip aggregate is NaN (same as the
    SQL `dense` CTE). Later hours are unknown and stay entirely NaN, so
    windows reaching into the horizon only average what was actually observed,
    and lags landing in the horizon are left for fill_missing.
    """
    print("  Computing lag / rolling features...")

    start = future["ts_hour"].min()
    hist = history_before(df_hist, start)
    last_observed = min(df_hist["ts_hour"].max(), start - pd.Timedelta(hours=1))

    raw_lag1h_cols = [c for c in RAW_LAG1H_COLS if c in hist.columns]
    value_cols = ["trip_count", "target_rph"] + raw_lag1h_cols

    grid_start = start - pd.Timedelta(hours=MAX_LOOKBACK_HOURS)
    dense = pd.MultiIndex.from_product(
        [future["PUZone"].unique(), pd.date_range(grid_start, future["ts_hour"].max(), freq="h")],
        names=["PUZone", "ts_hour"],
    ).to_frame(index=False)
    dense = dense.merge(
        hist.loc[hist["ts_hour"] >= grid_start, ["PUZone", "ts_hour"] + value_cols],
        on=["PUZone", "ts_hour"], how="left",
    )
    observed = dense["ts_hour"] <= last_observed
    dense.loc[observed, "trip_count"] = dense.loc[observed, "trip_count"].fillna(0)

    def by_zone(col):
        return dense.groupby("PUZone", sort=False, dropna=False)[col]

    def lag(col, hours):
        return by_zone(col).shift(hours)

    def rolling(col, hours, stat="mean", min_periods=1):
        # Window is [t - hours, t - 1] — the current hour is never included
        return by_zone(col).transform(
            lambda s: getattr(s.shift(1).rolling(hours, min_periods=min_periods), stat)()
        )

    def positive(s):
        return s.where(s > 0)

    feats = dense[["PUZone", "ts_hour"]].copy()

    # Trip count lags
    feats["lag_1h_trip_count"]  = lag("trip_count", 1)
    feats["lag_2h_trip_count"]  = lag("trip_count", 2)
    feats["lag_24h_trip_count"] = lag("trip_count", 24)

    # RPH lags
    feats["lag_1h_rph"]   = lag("target_rph", 1)
    feats["lag_2h_rph"]   = lag("target_rph", 2)
    feats["lag_24h_rph"]  = lag("target_rph", 24)
    feats["lag_168h_rph"] = lag("target_rph", 168)

    # Rolling windows
    feats["rolling_24h_trip_mean"]  = rolling("trip_count", 24)
    feats["rolling_168h_trip_mean"] = rolling("trip_count", 168)
    feats["rolling_24h_rph_mean"]   = rolling("target_rph", 24)
    feats["rolling_7d_rph_mean"]    = rolling("target_rph", 168)
    feats["rolling_24h_rph_std"]    = rolling("target_rph", 24, "std", min_periods=2)
    if "frac_airport_pickup" in dense.columns:
        feats["rolling_airport_activity"] = rolling("frac_airport_pickup", 24)

    # Same-hour aggregates -> lag1h_* (mirrors the `shifted` CTE).
    # demand_vs_baseline is recomputed on the grid, not read from the store,
    # so an hour without pickups gives 0 exactly as it does in SQL.
    dense["demand_vs_baseline"] = dense["trip_count"] / positive(feats["rolling_24h_trip_mean"])
    for c in raw_lag1h_cols + ["demand_vs_baseline"]:
        feats[f"lag1h_{c}"] = lag(c, 1)

    # Derived — built only from lags / windows, never the current hour
    feats["trip_count_delta_1h"] = feats["lag_1h_trip_count"] - feats["lag_2h_trip_count"]
    feats["rph_delta_1h"]        = feats["lag_1h_rph"] - feats["lag_2h_rph"]
    feats["rph_zscore"] = (
        (feats["lag_1h_rph"] - feats["rolling_24h_rph_mean"])
        / positive(feats["rolling_24h_rph_std"])
    )
    feats["rph_cv"] = feats["rolling_24h_rph_std"] / positive(feats["rolling_24h_rph_mean"])
    feats["relative_zone_activity"] = (
        feats["lag_1h_trip_count"] / positive(feats["rolling_168h_trip_mean"])
    )

    return future.merge(feats, on=["PUZone", "ts_hour"], how="left")


# FILL MISSING VALUES
# Use per-zone medians (from history) rather than global column medians.
# Global median fill biases predictions toward average-zone behaviour;
# zone-level fill preserves the distribution each zone's model learned from.
def fill_missing(future, df_hist, features):
    print("  Filling missing values...")

    # Build per-zone medians for every model feature that exists in history
    hist_features = [c for c in features if c in df_hist.columns]
    zone_medians = (
        df_hist.groupby("PUZone")[hist_features]
        .median()
        .reset_index()
        .rename(columns={c: f"__med_{c}" for c in hist_features})
    )
    future = future.merge(zone_medians, on="PUZone", how="left")

    for col in features:
        if col not in future.columns:
            future[col] = np.nan

        null_mask = future[col].isna()
        if null_mask.any():
            med_col = f"__med_{col}"
            if med_col in future.columns:
                # fill with per-zone median
                future.loc[null_mask, col] = future.loc[null_mask, med_col]
            # any remaining NaNs (zones absent from history) → global median
            still_null = future[col].isna()
            if still_null.any():
                global_med = df_hist[col].median() if col in df_hist.columns else 0.0
                future[col] = future[col].fillna(global_med if not np.isnan(global_med) else 0.0)

    # Drop helper columns
    future = future.drop(columns=[c for c in future.columns if c.startswith("__med_")])
    return future


# RUN INFERENCE THROUGH ALL 4 LAYERS
def run_inference(future, models, features, df_hist):
    print("Running inference...")

    future = fill_missing(future, df_hist, features)

    X = future[features].copy()

    # Layer 1 — Demand
    future["pred_demand"] = models["demand"].predict(X)

    # Layer 2 — Profitability
    future["pred_rph"] = models["profit"].predict(X)

    # Layer 3 — Competition pressure
    future["pred_competition"] = models["competition"].predict(X)

    # Layer 4 — Tip probability (uses imputer fitted during training)
    X_tip = models["tip_imputer"].transform(X)
    future["pred_tip_prob"] = models["tip"].predict_proba(X_tip)[:, 1]

    return future


# SCORE & RANK
# Normalise each signal to [0, 1] then combine with SCORE_WEIGHTS.
def score_and_rank(future):
    print("Scoring opportunities...")

    def minmax(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)

    # Competition: lower raw score → better opportunity → invert after normalising
    future["score_demand"]      = minmax(future["pred_demand"])
    future["score_profit"]      = minmax(future["pred_rph"])
    future["score_competition"] = 1 - minmax(future["pred_competition"])
    future["score_tip"]         = minmax(future["pred_tip_prob"])

    future["opportunity_score"] = (
        SCORE_WEIGHTS["demand"]      * future["score_demand"]
        + SCORE_WEIGHTS["profit"]    * future["score_profit"]
        + SCORE_WEIGHTS["competition"] * future["score_competition"]
        + SCORE_WEIGHTS["tip"]       * future["score_tip"]
    )

    future["rank"] = (
        future.groupby("ts_hour")["opportunity_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    return future.sort_values(["ts_hour", "rank"])


# SAVE OUTPUT
OUTPUT_COLS = [
    "ts_hour", "PUZone", "PUBorough",
    "rank", "opportunity_score",
    "pred_demand", "pred_rph", "pred_competition", "pred_tip_prob",
    "score_demand", "score_profit", "score_competition", "score_tip",
]

def save_forecast(df_out, n_hours, top_n, run_metadata: dict):
    if top_n:
        df_out = df_out[df_out["rank"] <= top_n]

    cols = [c for c in OUTPUT_COLS if c in df_out.columns]
    df_out = df_out[cols].round(4)

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUTPUT_DIR / f"forecast_{n_hours}h_{ts_tag}.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n✔ Saved → {out_csv}  ({len(df_out):,} rows)")

    meta_path = OUTPUT_DIR / f"forecast_{n_hours}h_{ts_tag}.meta.json"
    meta_path.write_text(json.dumps(run_metadata, indent=2, default=str), encoding="utf-8")

    csv_key = f"{OUTPUT_S3_PREFIX}forecast_{n_hours}h_{ts_tag}.csv"
    meta_key = f"{OUTPUT_S3_PREFIX}forecast_{n_hours}h_{ts_tag}.meta.json"
    with open(out_csv, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, csv_key)
    with open(meta_path, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, meta_key)
    print(f" Uploaded → s3://{S3_BUCKET}/{csv_key}")
    print(f" Metadata → s3://{S3_BUCKET}/{meta_key}")

    return out_csv, meta_path, df_out


# PREVIEW
def print_preview(df_out, n=15):
    print(f"\n{'─'*72}")
    print(f"  TOP OPPORTUNITIES  (first hour shown)")
    print(f"{'─'*72}")
    first_hour = df_out["ts_hour"].min()
    preview = df_out[df_out["ts_hour"] == first_hour].head(n)
    print(preview[[
        "rank", "PUZone", "PUBorough",
        "opportunity_score", "pred_demand", "pred_rph", "pred_tip_prob"
    ]].to_string(index=False))
    print(f"{'─'*72}\n")


# MAIN
def main():
    parser = argparse.ArgumentParser(description="NYC Taxi Forecast")
    parser.add_argument("--hours", type=int,  default=24,
                        help="Number of future hours to forecast (default: 24)")
    parser.add_argument("--top",   type=int,  default=None,
                        help="Keep only top-N zones per hour (default: all)")
    parser.add_argument("--start", type=str,  default=None,
                        help="Forecast start datetime, e.g. '2024-06-01 08:00'")
    args = parser.parse_args()

    print(f"\n{'═'*72}")
    print(f"  NYC TAXI INTELLIGENCE ENGINE — FORECAST")
    print(f"  Horizon : {args.hours}h")
    print(f"{'═'*72}\n")

    models, features = load_models()
    df_hist = load_history(models["zone_enc"], models["borough_enc"])

    start_dt = resolve_forecast_start(df_hist, args.start)

    # attach_lag_features gets the full store (it cuts at start_dt itself);
    # everything else only ever sees history before the forecast start.
    store = df_hist
    df_hist = history_before(store, start_dt)

    future = build_future_frame(df_hist, start_dt, args.hours, models["zone_enc"])
    future = attach_lag_features(future, store)
    future = run_inference(future, models, features, df_hist)
    future = score_and_rank(future)

    assert_forecast_window(future, start_dt, args.hours)

    run_metadata = {
        "data_s3_key": DATA_S3_KEY,
        "data_min_ts_hour": str(pd.Timestamp(df_hist["ts_hour"].min())),
        "data_max_ts_hour": str(pd.Timestamp(df_hist["ts_hour"].max())),
        "forecast_start": str(pd.Timestamp(start_dt)),
        "forecast_end": str(
            pd.Timestamp(start_dt) + pd.Timedelta(hours=args.hours - 1)
        ),
        "forecast_hours": args.hours,
        "top_n": args.top,
        "start_mode": "manual" if args.start else "auto_last_hour_plus_one",
    }

    with mlflow.start_run(run_name=f"taxi_forecast_{args.hours}h"):
        mlflow.set_tag("run_type", "forecast")
        mlflow.log_param("forecast_hours", args.hours)
        mlflow.log_param("top_n", args.top)
        mlflow.log_param("start_override", args.start)
        mlflow.log_param("forecast_start", str(start_dt))
        mlflow.log_param("forecast_end", str(start_dt + pd.Timedelta(hours=args.hours - 1)))
        mlflow.log_param("data_min_ts_hour", run_metadata["data_min_ts_hour"])
        mlflow.log_param("data_max_ts_hour", run_metadata["data_max_ts_hour"])

        out_csv, meta_path, df_out = save_forecast(future, args.hours, args.top, run_metadata)
        mlflow.log_metric("forecast_rows", len(df_out))
        mlflow.log_artifact(str(out_csv))
        mlflow.log_artifact(str(meta_path))

    print_preview(df_out)


if __name__ == "__main__":
    main()