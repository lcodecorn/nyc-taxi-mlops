"""
drift.py — Monthly drift check for the zone-hour feature store.

Compares the month that was just ingested (TARGET_MONTH rows of
final/hourly_zone_timeseries.parquet) against the reference snapshot of the
data the production models were trained on (models/drift_reference.parquet +
models/drift_baseline.json, written by models/*/train.py via save_reference).

───────────────────────────────────────────────────────────────────────────────

SIGNALS

1. Feature drift
    PSI per feature (decision) + KS statistic (supporting evidence only —
    with hundreds of thousands of rows the KS p-value is ~0 for any trivial
    shift). Aggregated into:
        drift_share           fraction of features with PSI > threshold
        weighted_drift_share  same, weighted by demand-model feature importance
    Only weighted_drift_share gates retraining, so the ~15 correlated
    lag/rolling features can't trigger a retrain on their own.

2. Target drift
    PSI on trip_count and target_rph.

3. Performance drift
    Forecasts previously written to forecasts/ whose window overlaps
    TARGET_MONTH are joined to the actuals that just landed; realised
    demand MAE / profit RMSE are compared to the validation metrics the
    production models were trained with.

Calendar features (month, quarter, week_of_year, day_of_year) and the label
encoded zones are excluded — a new month always differs on those by
construction. Seasonal features (weather) are handled by comparing against
the same calendar month of the reference when it holds enough rows.

───────────────────────────────────────────────────────────────────────────────

OUTPUTS

    s3://$S3_BUCKET/monitoring/drift_<month>.json          summary + decision
    s3://$S3_BUCKET/monitoring/drift_<month>_features.csv  per-feature table
    MLflow experiment "taxi_drift_monitoring"              metrics + both files

Usage:
    TARGET_MONTH=2024-06 python monitoring/drift.py
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from scipy.stats import ks_2samp

load_dotenv()

# Config
DATA_S3_KEY = "final/hourly_zone_timeseries.parquet"
REFERENCE_S3_KEY = "models/drift_reference.parquet"
BASELINE_S3_KEY = "models/drift_baseline.json"
FORECAST_S3_PREFIX = "forecasts/"
REPORT_S3_PREFIX = "monitoring/"
TMP_TIMESERIES = "/tmp/drift_hourly_zone_timeseries.parquet"
MLFLOW_EXPERIMENT = "taxi_drift_monitoring"

REFERENCE_SAMPLE_ROWS = 200_000
MIN_SEASONAL_ROWS = 10_000      # same-month reference rows needed for a seasonal comparison
MIN_PERF_ROWS = 500             # matched forecast/actual rows needed to trust realised metrics
CATEGORICAL_MAX_UNIQUE = 20     # features with <= this many values use frequency PSI
PSI_BINS = 10
PSI_EPS = 1e-4

PSI_THRESHOLD = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))
DRIFT_SHARE_THRESHOLD = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.3"))
TARGET_PSI_THRESHOLD = float(os.getenv("DRIFT_TARGET_PSI_THRESHOLD", "0.2"))
PERF_TOLERANCE = float(os.getenv("DRIFT_PERF_TOLERANCE", "0.2"))   # 0.2 → 20% worse than validation

EXCLUDED_FEATURES = {
    "PUZone_enc", "PUBorough_enc",
    "month", "quarter", "week_of_year", "day_of_year",
}
TARGETS = ["trip_count", "target_rph"]

# The feature store keeps every zone-hour with at least one pickup (the
# forecasters need them to rebuild lag/rolling features), but models are
# trained — and drift is measured — only on zone-hours with this many trips.
MIN_TRIP_COUNT = 5


# Reference snapshot (called from models/*/train.py)
def save_reference(
    train_df: pd.DataFrame,
    features: list[str],
    demand_model,
    valid_metrics: dict[str, float],
    out_dir: Path,
) -> None:
    """
    Write drift_reference.parquet (sampled training rows) and
    drift_baseline.json (training window, feature importances, validation
    metrics) into out_dir, next to the model artifacts they describe.
    """
    drift_features = [f for f in features if f not in EXCLUDED_FEATURES]

    ref = train_df[["ts_hour", "PUZone", *drift_features, *TARGETS]]
    if len(ref) > REFERENCE_SAMPLE_ROWS:
        ref = ref.sample(REFERENCE_SAMPLE_ROWS, random_state=42)
    ref.to_parquet(out_dir / "drift_reference.parquet", index=False)

    importances = dict(zip(features, (float(v) for v in demand_model.feature_importances_)))
    baseline = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "train_start": str(train_df["ts_hour"].min()),
        "train_end": str(train_df["ts_hour"].max()),
        "train_rows": len(train_df),
        "features": drift_features,
        "feature_importance": {f: importances.get(f, 0.0) for f in drift_features},
        "valid_metrics": {k: float(v) for k, v in valid_metrics.items()},
    }
    (out_dir / "drift_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")


# Statistics
def _psi_from_props(ref_props: np.ndarray, cur_props: np.ndarray) -> float:
    r = np.clip(ref_props, PSI_EPS, None)
    c = np.clip(cur_props, PSI_EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def frequency_psi(ref: pd.Series, cur: pd.Series) -> float:
    """PSI over category frequencies (zones, binary flags, hour of day, …)."""
    r = ref.dropna().value_counts(normalize=True)
    c = cur.dropna().value_counts(normalize=True)
    if r.empty or c.empty:
        return float("nan")
    r, c = r.align(c, fill_value=0.0)
    return _psi_from_props(r.to_numpy(), c.to_numpy())


def psi(ref: pd.Series, cur: pd.Series) -> float:
    """PSI with quantile bins taken from the reference distribution."""
    ref = ref.dropna()
    cur = cur.dropna()
    if ref.empty or cur.empty:
        return float("nan")
    if ref.nunique() <= CATEGORICAL_MAX_UNIQUE:
        return frequency_psi(ref, cur)

    r = ref.to_numpy(dtype=float)
    c = cur.to_numpy(dtype=float)
    inner_edges = np.unique(np.quantile(r, np.linspace(0, 1, PSI_BINS + 1)[1:-1]))
    n_bins = len(inner_edges) + 1
    r_props = np.bincount(np.searchsorted(inner_edges, r, side="right"), minlength=n_bins) / len(r)
    c_props = np.bincount(np.searchsorted(inner_edges, c, side="right"), minlength=n_bins) / len(c)
    return _psi_from_props(r_props, c_props)


def feature_drift(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    features: list[str],
    importance: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for f in features:
        r, c = ref[f], cur[f]
        r_vals = r.dropna().to_numpy(dtype=float)
        c_vals = c.dropna().to_numpy(dtype=float)
        ks = ks_2samp(r_vals, c_vals) if len(r_vals) and len(c_vals) else None
        p = psi(r, c)
        rows.append({
            "feature": f,
            "psi": p,
            "drifted": bool(p > PSI_THRESHOLD),
            "ks_stat": float(ks.statistic) if ks else np.nan,
            "ks_pvalue": float(ks.pvalue) if ks else np.nan,
            "ref_mean": float(np.mean(r_vals)) if len(r_vals) else np.nan,
            "cur_mean": float(np.mean(c_vals)) if len(c_vals) else np.nan,
            "ref_null_rate": float(r.isna().mean()),
            "cur_null_rate": float(c.isna().mean()),
            "importance": float(importance.get(f, 0.0)),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def drift_shares(table: pd.DataFrame) -> tuple[float, float]:
    """(drift_share, importance-weighted drift_share)."""
    if table.empty:
        return 0.0, 0.0
    drifted = table["drifted"].astype(float)
    weights = table["importance"].clip(lower=0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=table.index)
    return float(drifted.mean()), float((drifted * weights).sum() / weights.sum())


def decide(
    weighted_drift_share: float,
    target_psi: dict[str, float],
    performance: dict | None,
) -> list[str]:
    """Return the list of retrain reasons; empty means the model is still fit for purpose."""
    reasons = []
    if weighted_drift_share > DRIFT_SHARE_THRESHOLD:
        reasons.append(
            f"feature drift: weighted drift share {weighted_drift_share:.2f} > {DRIFT_SHARE_THRESHOLD}"
        )
    for target, value in target_psi.items():
        if value > TARGET_PSI_THRESHOLD:
            reasons.append(f"target drift: PSI({target}) {value:.3f} > {TARGET_PSI_THRESHOLD}")
    if performance:
        for name in ("demand_mae", "profit_rmse"):
            ratio = performance.get(f"{name}_ratio")
            if ratio is not None and ratio > 1 + PERF_TOLERANCE:
                reasons.append(
                    f"performance drift: realised {name} is {ratio:.2f}x the validation value"
                )
    return reasons


# S3 I/O
def _s3():
    import boto3
    return boto3.client("s3")


def _load_baseline(s3, bucket: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=BASELINE_S3_KEY)
    except s3.exceptions.NoSuchKey:
        return None
    return json.loads(obj["Body"].read())


def _load_reference(s3, bucket: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=REFERENCE_S3_KEY)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _load_current_month(
    s3, bucket: str, columns: list[str], month_start: pd.Timestamp, month_end: pd.Timestamp
) -> pd.DataFrame:
    """Read only TARGET_MONTH rows (and only the needed columns) from the feature store,
    restricted to the same trip_count >= MIN_TRIP_COUNT population the models train on."""
    print(f"Downloading s3://{bucket}/{DATA_S3_KEY} ...")
    s3.download_file(bucket, DATA_S3_KEY, TMP_TIMESERIES)
    available = set(pq.read_schema(TMP_TIMESERIES).names)
    return pd.read_parquet(
        TMP_TIMESERIES,
        columns=[c for c in columns if c in available],
        filters=[
            ("ts_hour", ">=", month_start.to_pydatetime()),
            ("ts_hour", "<", month_end.to_pydatetime()),
            ("trip_count", ">=", MIN_TRIP_COUNT),
        ],
    )


def forecast_window(meta: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    First and last forecast hour from a forecast .meta.json. Older EC2
    forecasts only recorded forecast_start + forecast_hours, so derive the
    end from those when forecast_end is missing.
    """
    start = pd.Timestamp(meta["forecast_start"])
    if "forecast_end" in meta:
        return start, pd.Timestamp(meta["forecast_end"])
    return start, start + pd.Timedelta(hours=int(meta["forecast_hours"]) - 1)


def realised_performance(
    s3,
    bucket: str,
    actuals: pd.DataFrame,
    baseline_metrics: dict[str, float],
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
) -> dict | None:
    """
    Score past forecasts covering TARGET_MONTH against the actuals that just
    landed. Only zone-hours present in the feature store (i.e. with at least
    one trip) can be matched, so this measures error where demand existed.
    """
    meta_keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=FORECAST_S3_PREFIX):
        meta_keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".meta.json")]

    frames = []
    for key in sorted(meta_keys):
        meta = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        start, end = forecast_window(meta)
        if end < month_start or start >= month_end:
            continue
        csv_key = key[: -len(".meta.json")] + ".csv"
        body = s3.get_object(Bucket=bucket, Key=csv_key)["Body"].read()
        frames.append(pd.read_csv(io.BytesIO(body), parse_dates=["ts_hour"]))

    if not frames:
        print("No forecasts overlap this month — skipping performance drift.")
        return None

    # Keys sort by creation timestamp, so keep the most recent forecast per zone-hour
    forecasts = pd.concat(frames).drop_duplicates(["ts_hour", "PUZone"], keep="last")
    matched = forecasts.merge(actuals[["ts_hour", "PUZone", *TARGETS]], on=["ts_hour", "PUZone"])
    matched = matched[(matched["ts_hour"] >= month_start) & (matched["ts_hour"] < month_end)]

    if len(matched) < MIN_PERF_ROWS:
        print(f"Only {len(matched)} forecast/actual matches (< {MIN_PERF_ROWS}) — skipping performance drift.")
        return None

    demand_mae = float((matched["pred_demand"] - matched["trip_count"]).abs().mean())
    rph = matched.dropna(subset=["target_rph", "pred_rph"])
    profit_rmse = float(np.sqrt(((rph["pred_rph"] - rph["target_rph"]) ** 2).mean()))

    result = {"matched_rows": len(matched), "demand_mae": demand_mae, "profit_rmse": profit_rmse}
    for name, base_key in (("demand_mae", "demand_valid_mae"), ("profit_rmse", "profit_valid_rmse")):
        base = baseline_metrics.get(base_key)
        if base:
            result[f"baseline_{name}"] = base
            result[f"{name}_ratio"] = result[name] / base
    return result


# Publishing
def _log_to_mlflow(report: dict, table: pd.DataFrame | None, paths: list[Path]) -> None:
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"drift_{report['month']}"):
        mlflow.set_tag("run_type", "drift_check")
        mlflow.set_tag("should_retrain", str(report["should_retrain"]))
        mlflow.log_params({
            "month": report["month"],
            "reference_mode": report.get("reference_mode"),
            "model_train_start": report.get("model_train_start"),
            "model_train_end": report.get("model_train_end"),
            "psi_threshold": PSI_THRESHOLD,
            "drift_share_threshold": DRIFT_SHARE_THRESHOLD,
            "target_psi_threshold": TARGET_PSI_THRESHOLD,
            "perf_tolerance": PERF_TOLERANCE,
        })

        metrics = {"should_retrain": float(report["should_retrain"])}
        for key in ("drift_share", "weighted_drift_share", "n_drifted", "zone_mix_psi", "current_rows"):
            if report.get(key) is not None:
                metrics[key] = report[key]
        for target, value in (report.get("target_psi") or {}).items():
            metrics[f"target_psi_{target}"] = value
        for key, value in (report.get("performance") or {}).items():
            metrics[f"perf_{key}"] = value
        if table is not None:
            metrics.update({f"psi_{r.feature}": r.psi for r in table.itertuples()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)})

        for path in paths:
            mlflow.log_artifact(str(path))


def _publish(s3, bucket: str, report: dict, table: pd.DataFrame | None) -> None:
    month = report["month"]
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / f"drift_{month}.json"]
        paths[0].write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        if table is not None:
            paths.append(Path(tmp) / f"drift_{month}_features.csv")
            table.round(6).to_csv(paths[1], index=False)

        for path in paths:
            s3.upload_file(str(path), bucket, REPORT_S3_PREFIX + path.name)
            print(f"  {path.name} → s3://{bucket}/{REPORT_S3_PREFIX}{path.name}")

        _log_to_mlflow(report, table, paths)


# Main
def run_drift_check(month: str) -> dict:
    """Run the drift check for month (YYYY-MM), publish the report, return it."""
    bucket = os.environ["S3_BUCKET"]
    s3 = _s3()
    month_start = pd.Timestamp(f"{month}-01")
    month_end = month_start + pd.offsets.MonthBegin(1)

    baseline = _load_baseline(s3, bucket)
    if baseline is None:
        report = {
            "month": month,
            "should_retrain": True,
            "reasons": ["no drift reference yet — no model has been trained with drift tracking"],
        }
        _publish(s3, bucket, report, None)
        return report

    reference = _load_reference(s3, bucket)
    current = _load_current_month(
        s3, bucket, ["ts_hour", "PUZone", *baseline["features"], *TARGETS], month_start, month_end
    )
    if current.empty:
        raise ValueError(f"No feature-store rows for {month} in s3://{bucket}/{DATA_S3_KEY}")

    seasonal = reference[pd.to_datetime(reference["ts_hour"]).dt.month == month_start.month]
    if len(seasonal) >= MIN_SEASONAL_ROWS:
        ref, reference_mode = seasonal, "same_month"
    else:
        ref, reference_mode = reference, "full"
    print(f"Reference: {len(ref):,} rows ({reference_mode})  |  current: {len(current):,} rows")

    features = [f for f in baseline["features"] if f in current.columns and f in ref.columns]
    table = feature_drift(ref, current, features, baseline.get("feature_importance", {}))
    drift_share, weighted_share = drift_shares(table)

    target_psi = {t: psi(ref[t], current[t]) for t in TARGETS if t in current.columns}
    performance = realised_performance(
        s3, bucket, current, baseline.get("valid_metrics", {}), month_start, month_end
    )

    reasons = decide(weighted_share, target_psi, performance)
    report = {
        "month": month,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "should_retrain": bool(reasons),
        "reasons": reasons,
        "model_train_start": baseline.get("train_start"),
        "model_train_end": baseline.get("train_end"),
        "reference_mode": reference_mode,
        "reference_rows": len(ref),
        "current_rows": len(current),
        "n_features": len(table),
        "n_drifted": int(table["drifted"].sum()),
        "drift_share": drift_share,
        "weighted_drift_share": weighted_share,
        "top_drifted": table.head(10)[["feature", "psi"]].to_dict(orient="records"),
        "zone_mix_psi": frequency_psi(ref["PUZone"], current["PUZone"]),
        "target_psi": target_psi,
        "performance": performance,
    }

    _publish(s3, bucket, report, table)
    return report


def main() -> None:
    month = os.getenv("TARGET_MONTH") or (datetime.utcnow().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    report = run_drift_check(month)

    print(f"\n{'═'*72}")
    print(f"  DRIFT CHECK — {month}")
    print(f"{'═'*72}")
    if "drift_share" in report:
        print(f"  drift share          : {report['drift_share']:.2f}")
        print(f"  weighted drift share : {report['weighted_drift_share']:.2f}")
        print(f"  target PSI           : {report['target_psi']}")
        print(f"  performance          : {report['performance']}")
    print(f"  should_retrain       : {report['should_retrain']}")
    for reason in report["reasons"]:
        print(f"    - {reason}")


if __name__ == "__main__":
    main()
