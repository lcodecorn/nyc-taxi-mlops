"""
Train/serve parity and leakage tests for the zone-hour lag / rolling features.

The hourly SQL in preprocess/features.py (training features) is run on
synthetic, deliberately gappy trips; the forecasters' attach_lag_features
(serving features) must reproduce it exactly from the stored rows alone,
using nothing at or after the forecast origin.

Pure in-memory / local temp files — no S3 / MLflow access, so these run in CI.
"""

import ast
import os
import re
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")  # boto3 client is built at import
from api import engine  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SERVING_FILES = ["api/engine.py", "models/local/forecast.py", "models/ec2/forecast.py"]

BUFFER_START = datetime(2025, 4, 1)
MONTH_START = datetime(2025, 5, 1)
MONTH_END = datetime(2025, 6, 1)

# zone -> probability an hour has any pickups (None = the unknown-location zone)
ZONE_ACTIVITY = {"Busy": 0.95, "Sparse": 0.35, "Rare": 0.06, None: 0.2}


def _synthetic_trips(rng) -> pd.DataFrame:
    hours = pd.date_range(BUFFER_START, MONTH_END, freq="h", inclusive="left")
    frames = []
    for zone, p_active in ZONE_ACTIVITY.items():
        active = hours[rng.random(len(hours)) < p_active]
        ts = active.repeat(rng.integers(1, 12, len(active)))
        n = len(ts)
        dur = rng.uniform(5, 60, n)
        rev = rng.uniform(8, 80, n)
        frames.append(pd.DataFrame({
            "PUZone": zone,
            "PUBorough": None if zone is None else "Manhattan",
            "DOZone": rng.choice(["Busy", "Sparse", "JFK Airport"], n),
            "ts_hour": ts,
            "service_date": ts.normalize(),
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "week_of_year": ts.isocalendar().week.to_numpy(),
            "month": ts.month,
            "quarter": ts.quarter,
            "year": ts.year,
            "day_of_year": ts.dayofyear,
            "is_weekend": (ts.dayofweek >= 5).astype(int),
            "is_rush": 0,
            "is_night": 0,
            "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
            "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
            "dow_sin": 0.0,
            "dow_cos": 0.0,
            "revenue_per_hour": rev / (dur / 60),
            "revenue": rev,
            "tip_amount": rng.uniform(0, 5, n),
            "trip_distance": rng.uniform(0.5, 15, n),
            "trip_duration_min": dur,
            "precipitation": rng.uniform(0, 1, n),
            "apparent_temperature": rng.uniform(0, 25, n),
            "wind_speed_10m": rng.uniform(0, 10, n),
            "is_raining": rng.random(n) < 0.2,
            "is_airport_pickup": rng.random(n) < 0.1,
            "is_airport_dropoff": rng.random(n) < 0.1,
        }))
    return pd.concat(frames, ignore_index=True)


def _hourly_sql(**fmt) -> str:
    """The hourly feature-store query from preprocess/features.py, formatted."""
    src = (REPO / "preprocess/features.py").read_text(encoding="utf-8")
    template = next(s for s in re.findall(r'con\.execute\(f"""(.*?)"""\)', src, re.S) if "WITH pickups AS" in s)
    return eval('f"""' + template + '"""', {}, fmt)


def _model_lag_features() -> list[str]:
    """Entries of train.py FEATURES that attach_lag_features is responsible for."""
    src = (REPO / "models/local/train.py").read_text(encoding="utf-8")
    features = ast.literal_eval(re.search(r"FEATURES = (\[.*?\])", src, re.S).group(1))
    return [f for f in features if f.startswith(("lag", "rolling", "rph_", "trip_count_", "relative_"))]


@pytest.fixture(scope="module")
def trips():
    return _synthetic_trips(np.random.default_rng(0))


@pytest.fixture(scope="module")
def store(trips, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("features")
    trips_path = (tmp / "features_month.parquet").as_posix()
    out_path = (tmp / "ts_month.parquet").as_posix()
    trips.to_parquet(trips_path, index=False)
    duckdb.connect().execute(_hourly_sql(
        TMP_FEATURES_MONTH=trips_path,
        TMP_TIMESERIES_MONTH=out_path,
        buffer_start=BUFFER_START,
        month_start=MONTH_START,
        month_end=MONTH_END,
    ))
    return pd.read_parquet(out_path)


@pytest.fixture(scope="module")
def origins():
    # >= 168h into the month so the full look-back is inside the stored month
    return pd.date_range(
        pd.Timestamp(MONTH_START) + pd.Timedelta(hours=200),
        pd.Timestamp(MONTH_END) - pd.Timedelta(hours=1),
        periods=25,
    ).floor("h")


def _future(store, start, hours=1):
    zones = store[["PUZone", "PUBorough"]].drop_duplicates()
    return pd.concat(
        [zones.assign(ts_hour=start + pd.Timedelta(hours=k)) for k in range(hours)],
        ignore_index=True,
    )


def test_store_is_one_row_per_zone_hour_with_pickups(store):
    assert store.duplicated(["PUZone", "ts_hour"]).sum() == 0
    assert store["trip_count"].min() >= 1
    assert store["ts_hour"].min() >= pd.Timestamp(MONTH_START)
    assert store["ts_hour"].max() < pd.Timestamp(MONTH_END)


def test_windows_span_hours_not_rows(trips, store):
    """A rare zone's 24h window must average 24 clock hours, empty ones as 0 trips."""
    hours = pd.date_range(BUFFER_START, MONTH_END, freq="h", inclusive="left")
    rare = (
        trips[trips["PUZone"] == "Rare"]
        .groupby("ts_hour")
        .agg(trip_count=("revenue", "size"), target_rph=("revenue_per_hour", "mean"))
        .reindex(hours)
    )
    rare["trip_count"] = rare["trip_count"].fillna(0)

    for _, row in store[store["PUZone"] == "Rare"].iterrows():
        t = row["ts_hour"]
        window = rare.loc[t - pd.Timedelta(hours=24): t - pd.Timedelta(hours=1)]
        assert len(window) == 24
        assert row["rolling_24h_trip_mean"] == pytest.approx(window["trip_count"].mean())
        expected_rph = window["target_rph"].mean()
        if np.isnan(expected_rph):
            assert pd.isna(row["rolling_24h_rph_mean"])
        else:
            assert row["rolling_24h_rph_mean"] == pytest.approx(expected_rph)
        assert row["lag_1h_trip_count"] == rare.loc[t - pd.Timedelta(hours=1), "trip_count"]


def test_rph_features_do_not_contain_the_target(store):
    reconstructed = store["lag_1h_rph"] + store["rph_delta_1h"]
    assert not np.isclose(reconstructed, store["target_rph"]).any()
    np.testing.assert_allclose(store["rph_delta_1h"], store["lag_1h_rph"] - store["lag_2h_rph"])

    std = store["rolling_24h_rph_std"]
    expected_z = (store["lag_1h_rph"] - store["rolling_24h_rph_mean"]) / std.where(std > 0)
    np.testing.assert_allclose(store["rph_zscore"], expected_z)


def test_serving_features_match_training_features(store, origins):
    features = _model_lag_features()
    for t in origins:
        served = engine.attach_lag_features(_future(store, t), store)
        truth = store[store["ts_hour"] == t]
        cmp = truth.merge(served, on=["PUZone", "ts_hour"], suffixes=("_sql", "_srv"))
        assert len(cmp) == len(truth)
        for f in features:
            np.testing.assert_allclose(
                cmp[f + "_srv"].astype(float), cmp[f + "_sql"].astype(float),
                rtol=1e-9, atol=1e-9, err_msg=f"{f} differs at origin {t}",
            )


def test_serving_never_reads_at_or_after_origin(store, origins):
    t = origins[5]
    future = _future(store, t, hours=24)
    clean = engine.attach_lag_features(future, store)

    poisoned = store.copy()
    numeric = poisoned.select_dtypes("number").columns
    poisoned.loc[poisoned["ts_hour"] >= t, numeric] = 1e9
    pd.testing.assert_frame_equal(clean, engine.attach_lag_features(future, poisoned))


def test_horizon_lags_are_unknown_not_zero(store, origins):
    t = origins[5]
    served = engine.attach_lag_features(_future(store, t, hours=24), store)
    later = served[served["ts_hour"] == t + pd.Timedelta(hours=5)]
    assert later["lag_1h_trip_count"].isna().all()
    assert later["lag_1h_rph"].isna().all()


def test_citywide_empty_hour_inside_store_is_zero_trips(store, origins):
    """An hour with no pickups in ANY zone is still an observed 0, not unknown."""
    t = origins[5]
    gappy = store[store["ts_hour"] != t - pd.Timedelta(hours=1)]
    served = engine.attach_lag_features(_future(gappy, t), gappy)
    assert (served["lag_1h_trip_count"] == 0).all()
    assert served["lag_1h_rph"].isna().all()


def test_hours_past_store_end_are_unknown(store):
    """Starting after the store ends: the missing hours in between are not 0 trips."""
    t = store["ts_hour"].max() + pd.Timedelta(hours=3)
    served = engine.attach_lag_features(_future(store, t), store)
    assert served["lag_1h_trip_count"].isna().all()
    assert served["lag_2h_trip_count"].isna().all()


def test_history_before_excludes_origin(store, origins):
    t = origins[0]
    assert engine.history_before(store, t)["ts_hour"].max() < t
    with pytest.raises(ValueError):
        engine.history_before(store, store["ts_hour"].min())


def _function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.body = [  # ignore progress print()s, which only the local CLI has
        s for s in node.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                and getattr(s.value.func, "id", "") == "print")
    ]
    return ast.dump(node)


@pytest.mark.parametrize("name", ["attach_lag_features", "history_before"])
def test_serving_copies_are_identical(name):
    """The API image only ships api/, so the logic is duplicated — keep it in lockstep."""
    assert len({_function_ast(REPO / p, name) for p in SERVING_FILES}) == 1


@pytest.mark.parametrize("const", ["RAW_LAG1H_COLS", "MAX_LOOKBACK_HOURS"])
def test_serving_constants_are_identical(const):
    pattern = re.compile(rf"^{const} = (\[.*?\]|\S+)", re.S | re.M)
    values = {pattern.search((REPO / p).read_text(encoding="utf-8")).group(1) for p in SERVING_FILES}
    assert len(values) == 1
