"""
features.py — Advanced feature engineering pipeline for NYC taxi
driver recommendation & forecasting system.

This pipeline builds a production-grade feature store for:

1. Demand forecasting
2. Revenue / profitability forecasting
3. Supply & competition estimation
4. Recommendation ranking systems

All computation runs entirely inside DuckDB.

───────────────────────────────────────────────────────────────────────────────

INPUT
    data/final/demand_dataset_enriched.parquet (full multi-month history)

OUTPUTS (S3 feature store, both append/upsert-by-month — see INCREMENTAL below)
    data/final/features.parquet
        Trip-level enriched features

    data/final/hourly_zone_timeseries.parquet
        Forecast-ready zone-hour feature store

───────────────────────────────────────────────────────────────────────────────

INCREMENTAL PROCESSING

    The upstream enriched dataset keeps growing every month (it holds the
    full retained history), but reprocessing all of it through the
    window-function pipeline below every run made peak memory grow with
    total history and eventually OOM-killed the task once retention passed
    ~40M rows.

    Instead, each run only reads the TARGET_MONTH plus a one calendar month
    lookback buffer (comfortably more than the 168h/7-day lag & rolling
    windows need) from the source parquet via a WHERE predicate, so DuckDB's
    working set is ~2 months of data no matter how many months are retained
    upstream. The buffer rows are used only to seed correct lag/rolling
    values at the start of the target month; only the target month's rows
    are written out. That freshly computed month is then upserted into the
    existing S3 feature-store files (any previous rows for that month are
    replaced, everything else untouched), so the two output files keep
    accumulating full history while the compute step itself stays flat.

    TARGET_MONTH env var (YYYY-MM) selects the month; defaults to the
    previous calendar month if unset.

───────────────────────────────────────────────────────────────────────────────

FEATURE GROUPS

✓ Revenue features
✓ Demand features
✓ Competition features
✓ Traffic / congestion features
✓ Weather features
✓ Airport features
✓ Lag features
✓ Rolling statistics
✓ Volatility indicators
✓ Momentum indicators
✓ Cyclical time encodings

───────────────────────────────────────────────────────────────────────────────

TARGET USE CASES

✓ XGBoost
✓ LightGBM
✓ CatBoost
✓ SARIMAX
✓ Prophet
✓ LSTM
✓ TFT
✓ Recommendation ranking systems
"""


import calendar
import os
from datetime import datetime, timedelta

import boto3
import duckdb
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()
S3_BUCKET = os.getenv("S3_BUCKET")
s3 = boto3.client("s3")

# S3 paths
INPUT_S3_KEY = "final/demand_dataset_enriched.parquet"
OUT_TRIPS_S3_KEY = "final/features.parquet"
OUT_TS_S3_KEY = "final/hourly_zone_timeseries.parquet"

# Local temp paths (DuckDB reads/writes these; then upload to S3)
TMP_INPUT = "/tmp/input.parquet"
TMP_FEATURES_MONTH = "/tmp/features_month.parquet"       # target month + buffer
TMP_TIMESERIES_MONTH = "/tmp/hourly_zone_timeseries_month.parquet"  # target month only
TMP_EXISTING = "/tmp/existing_store.parquet"
TMP_MERGED = "/tmp/merged_store.parquet"

LOOKBACK_MONTHS = 1  # >> the 168h/7-day lag & rolling windows; cheap insurance


def _resolve_target_month() -> str:
    raw = os.getenv("TARGET_MONTH", "").strip()
    if raw:
        datetime.strptime(raw, "%Y-%m")  # validate format, raise if malformed
        return raw
    prev_month_last_day = datetime.utcnow().replace(day=1) - timedelta(days=1)
    return prev_month_last_day.strftime("%Y-%m")


def _add_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    return dt.replace(year=year, month=month)


TARGET_MONTH = _resolve_target_month()
month_start = datetime.strptime(TARGET_MONTH, "%Y-%m")
days_in_month = calendar.monthrange(month_start.year, month_start.month)[1]
month_end = month_start + timedelta(days=days_in_month)  # exclusive
buffer_start = _add_months(month_start, -LOOKBACK_MONTHS)

print(
    f"Target month: {TARGET_MONTH}  "
    f"(reading {buffer_start:%Y-%m-%d} -> {month_end:%Y-%m-%d}, "
    f"writing rows >= {month_start:%Y-%m-%d})"
)

obj = s3.get_object(Bucket=S3_BUCKET, Key=INPUT_S3_KEY)
with open(TMP_INPUT, "wb") as f:
    f.write(obj["Body"].read())

# DuckDB setup. Even bounded to ~2 months of data, cap memory and let DuckDB
# spill to disk instead of getting OOM-killed if a month is unexpectedly big.
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3GB'")
con.execute("PRAGMA temp_directory='/tmp/duckdb_spill'")

MONTH_WINDOW_FILTER = f"""
    tpep_pickup_datetime >= TIMESTAMP '{buffer_start:%Y-%m-%d}'
    AND tpep_pickup_datetime < TIMESTAMP '{month_end:%Y-%m-%d}'
"""

# Input row count actually loaded for this run (buffer + target month only,
# not the full retained history)
n_total = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{TMP_INPUT}')
    WHERE {MONTH_WINDOW_FILTER}
""").fetchone()[0]
print(f"\nInput rows for {TARGET_MONTH} window: {n_total:,}")

# 1. TRIP-LEVEL FEATURE ENGINEERING (buffer + target month, so lag/rolling
#    context is available; only target-month rows get persisted downstream)
print("\nBuilding trip-level features...")
con.execute(f"""
COPY (
    WITH base AS (
        SELECT *,
            epoch(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0 AS trip_duration_min,
            GREATEST(fare_amount + extra + tip_amount + tolls_amount + Airport_fee, 0) AS revenue
        FROM read_parquet('{TMP_INPUT}')
        WHERE {MONTH_WINDOW_FILTER}
    ),
    features AS (
        SELECT *,
            CASE WHEN trip_duration_min > 1 THEN revenue / (trip_duration_min / 60.0) ELSE NULL END AS revenue_per_hour,
            date_trunc('hour', tpep_pickup_datetime) AS ts_hour,
            CAST(date_trunc('day', tpep_pickup_datetime) AS DATE) AS service_date,
            hour(tpep_pickup_datetime) AS hour,
            dayofweek(tpep_pickup_datetime) AS day_of_week,
            weekofyear(tpep_pickup_datetime) AS week_of_year,
            month(tpep_pickup_datetime) AS month,
            quarter(tpep_pickup_datetime) AS quarter,
            year(tpep_pickup_datetime) AS year,
            dayofyear(tpep_pickup_datetime) AS day_of_year,
            CASE WHEN dayofweek(tpep_pickup_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
            CASE WHEN hour(tpep_pickup_datetime) BETWEEN 7 AND 9 THEN 1 ELSE 0 END AS is_am_rush,
            CASE WHEN hour(tpep_pickup_datetime) BETWEEN 17 AND 19 THEN 1 ELSE 0 END AS is_pm_rush,
            CASE WHEN hour(tpep_pickup_datetime) BETWEEN 7 AND 9 OR hour(tpep_pickup_datetime) BETWEEN 17 AND 19 THEN 1 ELSE 0 END AS is_rush,
            CASE WHEN hour(tpep_pickup_datetime) NOT BETWEEN 6 AND 21 THEN 1 ELSE 0 END AS is_night,
            SIN(2 * PI() * hour(tpep_pickup_datetime) / 24.0) AS hour_sin,
            COS(2 * PI() * hour(tpep_pickup_datetime) / 24.0) AS hour_cos,
            SIN(2 * PI() * dayofweek(tpep_pickup_datetime) / 7.0) AS dow_sin,
            COS(2 * PI() * dayofweek(tpep_pickup_datetime) / 7.0) AS dow_cos,
            CASE WHEN precipitation > 0 THEN 1 ELSE 0 END AS is_raining,
            CASE WHEN apparent_temperature < 5 THEN 1 ELSE 0 END AS is_cold,
            CASE WHEN apparent_temperature > 30 THEN 1 ELSE 0 END AS is_hot,
            CASE WHEN PUZone IN ('JFK Airport', 'LaGuardia Airport', 'Newark Airport') THEN 1 ELSE 0 END AS is_airport_pickup,
            CASE WHEN DOZone IN ('JFK Airport', 'LaGuardia Airport', 'Newark Airport') THEN 1 ELSE 0 END AS is_airport_dropoff
        FROM base
    )
    SELECT * FROM features
    WHERE trip_duration_min BETWEEN 2 AND 180
      AND trip_distance BETWEEN 0.1 AND 100
      AND revenue BETWEEN 3 AND 500
      AND revenue_per_hour BETWEEN 5 AND 500
) TO '{TMP_FEATURES_MONTH}' (FORMAT PARQUET)
""")

# 2. HOURLY ZONE FEATURE STORE (computed over buffer + target month so
#    lag/rolling windows are correct at the start of the target month; only
#    target-month rows are kept in the final SELECT below)
print("\nBuilding hourly forecasting feature store...")

con.execute(f"""
COPY (

    --------------------------------------------------------------------------
    -- Pickup aggregation
    --------------------------------------------------------------------------

    WITH pickups AS (

        SELECT

            ------------------------------------------------------------------
            -- Keys
            ------------------------------------------------------------------

            PUZone,
            PUBorough,
            ts_hour,
            service_date,

            ------------------------------------------------------------------
            -- Calendar
            ------------------------------------------------------------------

            hour,
            day_of_week,
            week_of_year,
            month,
            quarter,
            year,
            day_of_year,

            is_weekend,
            is_rush,
            is_night,

            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,

            ------------------------------------------------------------------
            -- Demand
            ------------------------------------------------------------------

            COUNT(*) AS trip_count,

            ------------------------------------------------------------------
            -- Revenue metrics
            ------------------------------------------------------------------

            AVG(revenue_per_hour)
                AS target_rph,

            MEDIAN(revenue_per_hour)
                AS median_rph,

            QUANTILE_CONT(revenue_per_hour, 0.75)
                AS p75_rph,

            AVG(revenue)
                AS avg_revenue,

            AVG(tip_amount)
                AS avg_tip,

            AVG(
                CASE
                    WHEN tip_amount > 0
                    THEN 1.0 ELSE 0.0
                END
            ) AS tip_rate,

            ------------------------------------------------------------------
            -- Trip metrics
            ------------------------------------------------------------------

            AVG(trip_distance)
                AS avg_distance_mi,

            AVG(trip_duration_min)
                AS avg_duration_min,

            ------------------------------------------------------------------
            -- Weather
            ------------------------------------------------------------------

            AVG(precipitation)
                AS precipitation,

            AVG(apparent_temperature)
                AS temperature,

            AVG(wind_speed_10m)
                AS wind_speed,

            AVG(is_raining::DOUBLE)
                AS frac_raining,

            ------------------------------------------------------------------
            -- Airport activity
            ------------------------------------------------------------------

            AVG(is_airport_pickup::DOUBLE)
                AS frac_airport_pickup,

            AVG(is_airport_dropoff::DOUBLE)
                AS frac_airport_dropoff

        FROM read_parquet('{TMP_FEATURES_MONTH}')

        GROUP BY
            PUZone,
            PUBorough,
            ts_hour,
            service_date,
            hour,
            day_of_week,
            week_of_year,
            month,
            quarter,
            year,
            day_of_year,
            is_weekend,
            is_rush,
            is_night,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos
    ),

    --------------------------------------------------------------------------
    -- Dropoff aggregation
    --------------------------------------------------------------------------

    dropoffs AS (

        SELECT
            DOZone AS zone,
            ts_hour,
            COUNT(*) AS dropoff_count

        FROM read_parquet('{TMP_FEATURES_MONTH}')

        GROUP BY
            DOZone,
            ts_hour
    ),

    --------------------------------------------------------------------------
    -- Merge pickups + dropoffs
    --------------------------------------------------------------------------

    merged AS (

        SELECT

            p.*,

            COALESCE(d.dropoff_count, 0)
                AS dropoff_count,

            ------------------------------------------------------------------
            -- Supply / competition features
            ------------------------------------------------------------------

            p.trip_count
            - COALESCE(d.dropoff_count, 0)
                AS net_pickup_flow,

            CASE
                WHEN COALESCE(d.dropoff_count, 0) > 0
                THEN p.trip_count::DOUBLE
                     / d.dropoff_count
                ELSE NULL
            END AS pickup_dropoff_ratio

        FROM pickups p

        LEFT JOIN dropoffs d
            ON p.PUZone = d.zone
           AND p.ts_hour = d.ts_hour
    ),

    --------------------------------------------------------------------------
    -- Gap-free zone × hour grid
    --
    -- merged only has a row for zone-hours that had pickups, so ROWS frames
    -- and LAG(n) over it count *rows*, not hours: for a quiet zone the "24h"
    -- window could reach back days and lag_1h could be yesterday. Every
    -- time-series feature below is computed on this grid instead, where each
    -- zone has exactly one row per hour, so N rows == N hours.
    --
    -- An hour with no pickups gets trip_count = 0 (that is an observation);
    -- every other column is a per-trip aggregate and stays NULL (nothing was
    -- observed; AVG / STDDEV_SAMP skip NULLs). That is exactly what can be
    -- rebuilt from the stored rows, which is how the forecasters recreate
    -- these features at serving time (attach_lag_features).
    --------------------------------------------------------------------------

    grid AS (

        SELECT
            z.PUZone,
            h.ts_hour

        FROM (SELECT DISTINCT PUZone FROM merged) z

        CROSS JOIN range(
            TIMESTAMP '{buffer_start:%Y-%m-%d}',
            TIMESTAMP '{month_end:%Y-%m-%d}',
            INTERVAL 1 HOUR
        ) h(ts_hour)
    ),

    dense AS (

        SELECT
            g.PUZone,
            g.ts_hour,

            COALESCE(m.trip_count, 0) AS trip_count,

            m.target_rph,
            m.avg_revenue,
            m.avg_tip,
            m.tip_rate,
            m.avg_distance_mi,
            m.avg_duration_min,
            m.precipitation,
            m.temperature,
            m.wind_speed,
            m.frac_raining,
            m.frac_airport_pickup,
            m.frac_airport_dropoff,
            m.dropoff_count,
            m.net_pickup_flow,

            ------------------------------------------------------------------
            -- Traffic speed estimate
            ------------------------------------------------------------------

            CASE
                WHEN m.avg_duration_min > 0
                THEN m.avg_distance_mi
                     / (m.avg_duration_min / 60.0)
                ELSE NULL
            END AS avg_speed_mph,

            ------------------------------------------------------------------
            -- Congestion score
            ------------------------------------------------------------------

            CASE
                WHEN m.avg_duration_min > 0
                     AND m.avg_distance_mi > 0
                THEN 1.0 /
                     (
                        m.avg_distance_mi
                        / (m.avg_duration_min / 60.0)
                     )
                ELSE NULL
            END AS congestion_score

        FROM grid g

        LEFT JOIN merged m
            ON m.PUZone IS NOT DISTINCT FROM g.PUZone
           AND m.ts_hour = g.ts_hour
    ),

    --------------------------------------------------------------------------
    -- Lag & rolling features (on the dense grid; the current hour is always
    -- excluded from every window)
    --------------------------------------------------------------------------

    lagged AS (

        SELECT
            *,

            ------------------------------------------------------------------
            -- Revenue lags
            ------------------------------------------------------------------

            LAG(target_rph, 1)   OVER w AS lag_1h_rph,
            LAG(target_rph, 2)   OVER w AS lag_2h_rph,
            LAG(target_rph, 24)  OVER w AS lag_24h_rph,
            LAG(target_rph, 168) OVER w AS lag_168h_rph,

            ------------------------------------------------------------------
            -- Demand lags
            ------------------------------------------------------------------

            LAG(trip_count, 1)    OVER w AS lag_1h_trip_count,
            LAG(trip_count, 2)    OVER w AS lag_2h_trip_count,
            LAG(trip_count, 24)   OVER w AS lag_24h_trip_count,
            LAG(dropoff_count, 1) OVER w AS lag_1h_dropoff_count,

            ------------------------------------------------------------------
            -- Rolling demand
            ------------------------------------------------------------------

            AVG(trip_count) OVER w24  AS rolling_24h_trip_mean,
            AVG(trip_count) OVER w168 AS rolling_168h_trip_mean,

            ------------------------------------------------------------------
            -- Rolling revenue
            ------------------------------------------------------------------

            AVG(target_rph) OVER w24  AS rolling_24h_rph_mean,
            AVG(target_rph) OVER w168 AS rolling_7d_rph_mean,

            ------------------------------------------------------------------
            -- Rolling volatility
            ------------------------------------------------------------------

            STDDEV_SAMP(target_rph) OVER w24 AS rolling_24h_rph_std,

            ------------------------------------------------------------------
            -- Rolling airport activity
            ------------------------------------------------------------------

            AVG(frac_airport_pickup) OVER w24 AS rolling_airport_activity,

            ------------------------------------------------------------------
            -- Weather momentum (same-hour; not a model input)
            ------------------------------------------------------------------

            precipitation - LAG(precipitation, 1) OVER w AS precipitation_delta

        FROM dense

        WINDOW
            w    AS (PARTITION BY PUZone ORDER BY ts_hour),
            w24  AS (PARTITION BY PUZone ORDER BY ts_hour
                     ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING),
            w168 AS (PARTITION BY PUZone ORDER BY ts_hour
                     ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING)
    ),

    --------------------------------------------------------------------------
    -- Advanced derived features
    --
    -- Anything that is a model input must only use hours BEFORE ts_hour, so
    -- the revenue momentum / surge features are built from lag_1h_rph, never
    -- target_rph. (They used to use target_rph: lag_1h_rph + rph_delta_1h
    -- reconstructed the profit model's label exactly.)
    --------------------------------------------------------------------------

    final_features AS (

        SELECT
            *,

            ------------------------------------------------------------------
            -- Demand momentum
            ------------------------------------------------------------------

            lag_1h_trip_count - lag_2h_trip_count
                AS trip_count_delta_1h,

            -- Uses this hour's trip_count, so models only ever see it as
            -- lag1h_demand_vs_baseline
            CASE
                WHEN rolling_24h_trip_mean > 0
                THEN trip_count
                     / rolling_24h_trip_mean
                ELSE NULL
            END AS demand_vs_baseline,

            ------------------------------------------------------------------
            -- Revenue momentum
            ------------------------------------------------------------------

            lag_1h_rph - lag_2h_rph
                AS rph_delta_1h,

            ------------------------------------------------------------------
            -- Revenue surge z-score (last hour vs. trailing 24h)
            ------------------------------------------------------------------

            CASE
                WHEN rolling_24h_rph_std > 0
                THEN (
                    lag_1h_rph
                    - rolling_24h_rph_mean
                ) / rolling_24h_rph_std
                ELSE NULL
            END AS rph_zscore,

            ------------------------------------------------------------------
            -- Revenue stability
            ------------------------------------------------------------------

            CASE
                WHEN rolling_24h_rph_mean > 0
                THEN rolling_24h_rph_std
                     / rolling_24h_rph_mean
                ELSE NULL
            END AS rph_cv,

            ------------------------------------------------------------------
            -- Relative zone activity
            ------------------------------------------------------------------

            CASE
                WHEN rolling_168h_trip_mean > 0
                THEN lag_1h_trip_count / rolling_168h_trip_mean
                ELSE NULL
            END AS relative_zone_activity

        FROM lagged
    ),

    --------------------------------------------------------------------------
    -- Shift same-hour aggregates back by 1h
    --
    -- avg_revenue, avg_tip, tip_rate, avg_distance_mi/duration/speed,
    -- congestion_score, dropoff_count, net_pickup_flow, frac_airport_*,
    -- precipitation/temperature/wind/frac_raining and demand_vs_baseline are
    -- all computed from trips that occur DURING ts_hour — the same hour as
    -- the trip_count / target_rph targets. A forecast made before ts_hour
    -- cannot know these values, so we expose lag1h_* versions (the prior
    -- hour's realized value) for use as model features.
    --------------------------------------------------------------------------

    shifted AS (
        SELECT
            *,
            LAG(frac_airport_pickup, 1)  OVER w AS lag1h_frac_airport_pickup,
            LAG(frac_airport_dropoff, 1) OVER w AS lag1h_frac_airport_dropoff,
            LAG(precipitation, 1)        OVER w AS lag1h_precipitation,
            LAG(temperature, 1)          OVER w AS lag1h_temperature,
            LAG(wind_speed, 1)           OVER w AS lag1h_wind_speed,
            LAG(frac_raining, 1)         OVER w AS lag1h_frac_raining,
            LAG(demand_vs_baseline, 1)   OVER w AS lag1h_demand_vs_baseline,
            LAG(dropoff_count, 1)        OVER w AS lag1h_dropoff_count,
            LAG(net_pickup_flow, 1)      OVER w AS lag1h_net_pickup_flow,
            LAG(avg_revenue, 1)          OVER w AS lag1h_avg_revenue,
            LAG(avg_tip, 1)              OVER w AS lag1h_avg_tip,
            LAG(tip_rate, 1)             OVER w AS lag1h_tip_rate,
            LAG(avg_distance_mi, 1)      OVER w AS lag1h_avg_distance_mi,
            LAG(avg_duration_min, 1)     OVER w AS lag1h_avg_duration_min,
            LAG(avg_speed_mph, 1)        OVER w AS lag1h_avg_speed_mph,
            LAG(congestion_score, 1)     OVER w AS lag1h_congestion_score
        FROM final_features
        WINDOW w AS (PARTITION BY PUZone ORDER BY ts_hour)
    )

    --------------------------------------------------------------------------
    -- Final dataset — back to one row per zone-hour WITH pickups; buffer
    -- rows dropped, only the target month is kept.
    --
    -- No minimum trip_count here: every zone-hour with a pickup is stored so
    -- the forecasters can rebuild the dense grid (and hence identical
    -- lag/rolling features) from this file. Training applies its own
    -- trip_count >= MIN_TRIP_COUNT filter (monitoring/drift.py).
    --------------------------------------------------------------------------

    SELECT
        m.*,
        s.* EXCLUDE (
            PUZone, ts_hour, trip_count, target_rph,
            avg_revenue, avg_tip, tip_rate,
            avg_distance_mi, avg_duration_min,
            precipitation, temperature, wind_speed, frac_raining,
            frac_airport_pickup, frac_airport_dropoff,
            dropoff_count, net_pickup_flow
        )

    FROM merged m

    JOIN shifted s
        ON s.PUZone IS NOT DISTINCT FROM m.PUZone
       AND s.ts_hour = m.ts_hour

    WHERE m.ts_hour >= TIMESTAMP '{month_start:%Y-%m-%d}'
      AND m.ts_hour <  TIMESTAMP '{month_end:%Y-%m-%d}'

    ORDER BY
        m.PUZone,
        m.ts_hour

) TO '{TMP_TIMESERIES_MONTH}' (FORMAT PARQUET)
""")


def upsert_month_to_s3(
    s3_key: str,
    new_month_local_path: str,
    month_col: str,
    extra_new_rows_filter: str = "TRUE",
) -> None:
    """Merge this run's freshly computed month into the existing S3 parquet.

    Any previously stored rows for [month_start, month_end) are dropped and
    replaced with the new ones (safe to re-run/backfill a month), everything
    else in the store is left untouched. Falls back to writing the month
    alone if the S3 object doesn't exist yet (first ever run).
    """
    month_filter = (
        f"{month_col} >= TIMESTAMP '{month_start:%Y-%m-%d}' "
        f"AND {month_col} < TIMESTAMP '{month_end:%Y-%m-%d}'"
    )

    existing_clause = None
    try:
        s3.download_file(S3_BUCKET, s3_key, TMP_EXISTING)
        existing_clause = f"""
            SELECT * FROM read_parquet('{TMP_EXISTING}')
            WHERE NOT ({month_filter})
        """
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
            raise
        print(f"No existing s3://{S3_BUCKET}/{s3_key} yet — writing first month.")

    new_rows_clause = f"""
        SELECT * FROM read_parquet('{new_month_local_path}')
        WHERE {month_filter} AND ({extra_new_rows_filter})
    """

    query = (
        f"COPY ({existing_clause} UNION ALL BY NAME {new_rows_clause}) TO '{TMP_MERGED}' (FORMAT PARQUET)"
        if existing_clause
        else f"COPY ({new_rows_clause}) TO '{TMP_MERGED}' (FORMAT PARQUET)"
    )
    con.execute(query)

    with open(TMP_MERGED, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, s3_key)


# Trip-level output: only the target month is persisted (buffer rows were
# only needed as context for the hourly window functions above).
upsert_month_to_s3(OUT_TRIPS_S3_KEY, TMP_FEATURES_MONTH, month_col="tpep_pickup_datetime")
print(f"Upserted {TARGET_MONTH} into s3://{S3_BUCKET}/{OUT_TRIPS_S3_KEY}")

# Hourly timeseries output (TMP_TIMESERIES_MONTH is already target-month-only).
upsert_month_to_s3(OUT_TS_S3_KEY, TMP_TIMESERIES_MONTH, month_col="ts_hour")
print(f"Upserted {TARGET_MONTH} into s3://{S3_BUCKET}/{OUT_TS_S3_KEY}")

# 3. DATASET SUMMARY (full feature store, post-upsert)
summary = con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT PUZone) AS zones,
        MIN(ts_hour) AS min_ts,
        MAX(ts_hour) AS max_ts,
        AVG(target_rph) AS avg_rph,
        AVG(trip_count) AS avg_trip_count
    FROM read_parquet('{TMP_MERGED}')
""").fetchdf()

print("\nDataset summary (full store after this run's upsert):")
print(summary.to_string(index=False))

# 4. PREVIEW (target month only)
preview = con.execute(f"""
    SELECT

        PUZone,
        ts_hour,

        ----------------------------------------------------------------------
        -- Demand
        ----------------------------------------------------------------------

        trip_count,
        dropoff_count,
        net_pickup_flow,

        ----------------------------------------------------------------------
        -- Revenue
        ----------------------------------------------------------------------

        target_rph,
        lag_1h_rph,
        rolling_24h_rph_mean,

        ----------------------------------------------------------------------
        -- Competition
        ----------------------------------------------------------------------

        pickup_dropoff_ratio,

        ----------------------------------------------------------------------
        -- Traffic
        ----------------------------------------------------------------------

        avg_speed_mph,
        congestion_score,

        ----------------------------------------------------------------------
        -- Volatility
        ----------------------------------------------------------------------

        rph_zscore,
        rph_cv,

        ----------------------------------------------------------------------
        -- Weather
        ----------------------------------------------------------------------

        precipitation,
        temperature

    FROM read_parquet('{TMP_TIMESERIES_MONTH}')

    ORDER BY ts_hour

    LIMIT 20
""").fetchdf()

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 50)

print("\nSample rows:\n")

print(
    preview.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\nDone.")

print("\nRecommended next steps:")
print("1. Train demand forecasting model")
print("2. Train revenue forecasting model")
print("3. Train competition/saturation model")
print("4. Build recommendation ranking engine")
print("5. Add neighboring zone spatial features")
