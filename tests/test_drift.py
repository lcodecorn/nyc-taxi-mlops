"""
Unit tests for monitoring/drift.py statistics and retrain decision.

Pure in-memory — no S3 / MLflow access, so these run in CI.
"""

import numpy as np
import pandas as pd
import pytest

from monitoring import drift


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_psi_identical_distribution_is_near_zero(rng):
    ref = pd.Series(rng.normal(0, 1, 50_000))
    cur = pd.Series(rng.normal(0, 1, 50_000))
    assert drift.psi(ref, cur) < 0.01


def test_psi_detects_mean_shift(rng):
    ref = pd.Series(rng.normal(0, 1, 50_000))
    cur = pd.Series(rng.normal(1, 1, 50_000))
    assert drift.psi(ref, cur) > drift.PSI_THRESHOLD


def test_psi_binary_feature_uses_frequencies():
    ref = pd.Series([0] * 900 + [1] * 100)
    same = pd.Series([0] * 450 + [1] * 50)
    flipped = pd.Series([0] * 100 + [1] * 900)
    assert drift.psi(ref, same) == pytest.approx(0.0, abs=1e-9)
    assert drift.psi(ref, flipped) > drift.PSI_THRESHOLD


def test_psi_ignores_nans_and_handles_empty():
    ref = pd.Series([np.nan] * 10 + list(range(100)), dtype=float)
    assert np.isfinite(drift.psi(ref, ref))
    assert np.isnan(drift.psi(ref, pd.Series([np.nan, np.nan])))


def test_frequency_psi_new_category_counts_as_drift():
    ref = pd.Series(["JFK Airport"] * 50 + ["Midtown Center"] * 50)
    cur = pd.Series(["Midtown Center"] * 50 + ["Astoria"] * 50)
    assert drift.frequency_psi(ref, cur) > drift.PSI_THRESHOLD


def test_drift_shares_weighted_by_importance():
    table = pd.DataFrame({
        "drifted": [True, False, False],
        "importance": [8.0, 1.0, 1.0],
    })
    share, weighted = drift.drift_shares(table)
    assert share == pytest.approx(1 / 3)
    assert weighted == pytest.approx(0.8)


def test_drift_shares_falls_back_to_equal_weights():
    table = pd.DataFrame({"drifted": [True, False], "importance": [0.0, 0.0]})
    assert drift.drift_shares(table) == (0.5, 0.5)


def test_feature_drift_flags_only_shifted_feature(rng):
    n = 20_000
    ref = pd.DataFrame({"stable": rng.normal(0, 1, n), "shifted": rng.normal(0, 1, n)})
    cur = pd.DataFrame({"stable": rng.normal(0, 1, n), "shifted": rng.normal(2, 1, n)})
    table = drift.feature_drift(ref, cur, ["stable", "shifted"], {"stable": 1, "shifted": 1})
    flagged = set(table.loc[table["drifted"], "feature"])
    assert flagged == {"shifted"}


def test_decide_no_drift():
    perf = {"demand_mae_ratio": 1.05, "profit_rmse_ratio": 0.9}
    assert drift.decide(0.1, {"trip_count": 0.05, "target_rph": 0.02}, perf) == []


def test_decide_collects_every_reason():
    perf = {"demand_mae_ratio": 1.5, "profit_rmse_ratio": 1.0}
    reasons = drift.decide(0.5, {"trip_count": 0.3, "target_rph": 0.01}, perf)
    assert len(reasons) == 3
    assert any(r.startswith("feature drift") for r in reasons)
    assert any("trip_count" in r for r in reasons)
    assert any("demand_mae" in r for r in reasons)


def test_decide_without_performance_data():
    assert drift.decide(0.0, {}, None) == []
