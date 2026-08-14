from __future__ import annotations

import random

import polars as pl
import pytest

from dataxid_profiling._alerts import Alert, AlertType, check_quality
from dataxid_profiling._analyzers import analyze
from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._correlations import compute_correlations
from dataxid_profiling._dataset_overview import compute_overview
from dataxid_profiling._type_inference import infer_types


def _get_alerts(df: pl.DataFrame, config: ProfileConfig | None = None) -> list[Alert]:
    config = config or ProfileConfig()
    column_types = infer_types(df, config)
    column_stats = analyze(df, column_types, config)
    overview = compute_overview(df, column_types, config)
    correlations = compute_correlations(df, column_types, config)
    return check_quality(column_stats, overview, config, correlations)


def _alert_types(alerts: list[Alert]) -> set[AlertType]:
    return {a.alert_type for a in alerts}


def _alerts_for_column(alerts: list[Alert], col: str) -> list[Alert]:
    return [a for a in alerts if a.column == col]


class TestHighMissing:
    def test_triggers_above_threshold(self):
        df = pl.DataFrame({"a": [1, None, None, None, None]})
        alerts = _get_alerts(df, ProfileConfig(missing_threshold=0.05))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.HIGH_MISSING for a in col_alerts)

    def test_no_alert_below_threshold(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, None]})
        alerts = _get_alerts(df, ProfileConfig(missing_threshold=0.5))
        col_alerts = _alerts_for_column(alerts, "a")
        assert not any(a.alert_type == AlertType.HIGH_MISSING for a in col_alerts)

    def test_missing_value_correct(self):
        df = pl.DataFrame({"a": [1, None, None, 4, 5]})
        alerts = _get_alerts(df, ProfileConfig(missing_threshold=0.05))
        missing_alerts = [
            a for a in alerts if a.alert_type == AlertType.HIGH_MISSING and a.column == "a"
        ]
        assert len(missing_alerts) == 1
        assert missing_alerts[0].value == pytest.approx(0.4)


class TestConstant:
    def test_single_value_triggers(self):
        df = pl.DataFrame({"a": [42, 42, 42, 42, 42]})
        alerts = _get_alerts(df)
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.CONSTANT for a in col_alerts)

    def test_multiple_values_no_alert(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        alerts = _get_alerts(df)
        col_alerts = _alerts_for_column(alerts, "a")
        assert not any(a.alert_type == AlertType.CONSTANT for a in col_alerts)


class TestHighCardinality:
    def test_numeric_high_cardinality(self):
        df = pl.DataFrame({"a": list(range(100))})
        alerts = _get_alerts(df, ProfileConfig(cardinality_threshold=0.95))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.HIGH_CARDINALITY for a in col_alerts)

    def test_categorical_high_cardinality(self):
        df = pl.DataFrame({"a": [f"val_{i}" for i in range(20)]})
        alerts = _get_alerts(df, ProfileConfig(cardinality_threshold=0.95, text_unique_ratio=1.0))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.HIGH_CARDINALITY for a in col_alerts)


class TestDuplicates:
    def test_duplicates_trigger(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        alerts = _get_alerts(df)
        assert any(a.alert_type == AlertType.DUPLICATES for a in alerts)

    def test_no_duplicates(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        alerts = _get_alerts(df)
        assert not any(a.alert_type == AlertType.DUPLICATES for a in alerts)

    def test_duplicate_is_dataset_level(self):
        df = pl.DataFrame({"a": [1, 1, 2]})
        dup_alerts = [a for a in _get_alerts(df) if a.alert_type == AlertType.DUPLICATES]
        assert all(a.column is None for a in dup_alerts)


class TestHighZeros:
    def test_zeros_trigger(self):
        df = pl.DataFrame({"a": [0, 0, 0, 1, 2]})
        alerts = _get_alerts(df, ProfileConfig(zero_threshold=0.05))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.HIGH_ZEROS for a in col_alerts)

    def test_no_zeros_no_alert(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        alerts = _get_alerts(df)
        col_alerts = _alerts_for_column(alerts, "a")
        assert not any(a.alert_type == AlertType.HIGH_ZEROS for a in col_alerts)


class TestSkewed:
    def test_skewed_triggers(self):
        # Highly right-skewed distribution
        df = pl.DataFrame({"a": [1] * 90 + [1000] * 10})
        alerts = _get_alerts(df, ProfileConfig(skewness_threshold=2.0))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.SKEWED for a in col_alerts)

    def test_symmetric_no_alert(self):
        df = pl.DataFrame({"a": list(range(1, 101))})
        alerts = _get_alerts(df, ProfileConfig(skewness_threshold=2.0))
        col_alerts = _alerts_for_column(alerts, "a")
        assert not any(a.alert_type == AlertType.SKEWED for a in col_alerts)


class TestImbalanced:
    def test_boolean_imbalanced(self):
        df = pl.DataFrame({"a": [True] * 95 + [False] * 5})
        alerts = _get_alerts(df, ProfileConfig(imbalance_threshold=0.9))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.IMBALANCED for a in col_alerts)

    def test_boolean_balanced_no_alert(self):
        df = pl.DataFrame({"a": [True, False, True, False]})
        alerts = _get_alerts(df, ProfileConfig(imbalance_threshold=0.9))
        col_alerts = _alerts_for_column(alerts, "a")
        assert not any(a.alert_type == AlertType.IMBALANCED for a in col_alerts)

    def test_categorical_imbalanced(self):
        df = pl.DataFrame({"a": ["x"] * 95 + ["y"] * 5})
        alerts = _get_alerts(df, ProfileConfig(imbalance_threshold=0.9, text_unique_ratio=1.0))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.IMBALANCED for a in col_alerts)


class TestHighCorrelation:
    def test_triggers_on_correlated_pair(self):
        n = 50
        x = list(range(n))
        df = pl.DataFrame({"a": x, "b": [v * 2 + 1 for v in x]})
        alerts = _get_alerts(df, ProfileConfig(correlation_threshold=0.8))
        corr_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_CORRELATION]
        assert len(corr_alerts) >= 1
        alert = corr_alerts[0]
        assert alert.value > 0.8
        assert "column_b" in alert.details
        assert "method" in alert.details

    def test_no_alert_below_threshold(self):
        import random

        random.seed(42)
        df = pl.DataFrame({"a": list(range(50)), "b": [random.random() for _ in range(50)]})
        alerts = _get_alerts(df, ProfileConfig(correlation_threshold=0.9))
        corr_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_CORRELATION]
        assert len(corr_alerts) == 0

    def test_details_has_method(self):
        n = 50
        x = list(range(n))
        df = pl.DataFrame({"a": x, "b": [v * 3 for v in x]})
        alerts = _get_alerts(df, ProfileConfig(correlation_threshold=0.5))
        corr_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_CORRELATION]
        assert len(corr_alerts) >= 1
        assert corr_alerts[0].details["method"] in ("phik", "pearson")

    def test_no_duplicate_pairs(self):
        n = 50
        x = list(range(n))
        df = pl.DataFrame({"a": x, "b": [v * 2 for v in x], "c": [v * 3 for v in x]})
        alerts = _get_alerts(df, ProfileConfig(correlation_threshold=0.5))
        corr_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_CORRELATION]
        pairs = {tuple(sorted((a.column, a.details["column_b"]))) for a in corr_alerts}
        assert len(pairs) == len(corr_alerts)

    def test_single_column_no_alert(self):
        df = pl.DataFrame({"a": list(range(10))})
        alerts = _get_alerts(df)
        corr_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_CORRELATION]
        assert len(corr_alerts) == 0


class TestUniform:
    def test_triggers_on_uniform_distribution(self):
        df = pl.DataFrame({"cat": ["A", "B", "C", "D"] * 25})
        alerts = _get_alerts(
            df, ProfileConfig(uniform_pvalue_threshold=0.05, text_unique_ratio=1.0)
        )
        uniform_alerts = [a for a in alerts if a.alert_type == AlertType.UNIFORM]
        assert len(uniform_alerts) == 1
        assert uniform_alerts[0].column == "cat"
        assert uniform_alerts[0].details["test"] == "chi2_gof"
        assert uniform_alerts[0].details["p_value"] > 0.05

    def test_no_alert_on_skewed_distribution(self):
        df = pl.DataFrame({"cat": ["A"] * 90 + ["B"] * 5 + ["C"] * 5})
        alerts = _get_alerts(
            df,
            ProfileConfig(
                uniform_pvalue_threshold=0.05,
                text_unique_ratio=1.0,
                imbalance_threshold=1.0,
            ),
        )
        uniform_alerts = [a for a in alerts if a.alert_type == AlertType.UNIFORM]
        assert len(uniform_alerts) == 0

    def test_details_contains_p_value(self):
        df = pl.DataFrame({"cat": ["X", "Y", "Z"] * 30})
        alerts = _get_alerts(
            df, ProfileConfig(uniform_pvalue_threshold=0.05, text_unique_ratio=1.0)
        )
        uniform_alerts = [a for a in alerts if a.alert_type == AlertType.UNIFORM]
        assert len(uniform_alerts) == 1
        assert "p_value" in uniform_alerts[0].details
        assert isinstance(uniform_alerts[0].details["p_value"], float)

    def test_single_category_no_alert(self):
        df = pl.DataFrame({"cat": ["A"] * 20})
        alerts = _get_alerts(df)
        uniform_alerts = [a for a in alerts if a.alert_type == AlertType.UNIFORM]
        assert len(uniform_alerts) == 0


class TestDatetimeAlerts:
    def test_unsorted_dates_alert(self):
        from datetime import datetime

        df = pl.DataFrame(
            {
                "ts": [
                    datetime(2024, 1, 3),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                ]
            }
        )
        alerts = _get_alerts(df)
        ts_alerts = _alerts_for_column(alerts, "ts")
        assert any(a.alert_type == AlertType.UNSORTED_DATES for a in ts_alerts)

    def test_irregular_intervals_alert(self):
        from datetime import datetime

        df = pl.DataFrame(
            {
                "ts": [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 1, 0),
                    datetime(2024, 1, 1, 5, 0),
                    datetime(2024, 1, 1, 6, 30),
                    datetime(2024, 1, 2, 0, 0),
                ]
            }
        )
        alerts = _get_alerts(df)
        ts_alerts = _alerts_for_column(alerts, "ts")
        assert any(a.alert_type == AlertType.IRREGULAR_INTERVALS for a in ts_alerts)
        irregular = [a for a in ts_alerts if a.alert_type == AlertType.IRREGULAR_INTERVALS]
        assert irregular[0].details["std_seconds"] > 0

    def test_large_gaps_alert(self):
        from datetime import datetime

        df = pl.DataFrame(
            {
                "ts": [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 1, 0),
                    datetime(2024, 1, 1, 2, 0),
                    datetime(2024, 1, 5, 0, 0),
                    datetime(2024, 1, 5, 1, 0),
                    datetime(2024, 1, 5, 2, 0),
                ]
            }
        )
        alerts = _get_alerts(df)
        ts_alerts = _alerts_for_column(alerts, "ts")
        assert any(a.alert_type == AlertType.LARGE_GAPS for a in ts_alerts)
        gap_alerts = [a for a in ts_alerts if a.alert_type == AlertType.LARGE_GAPS]
        assert gap_alerts[0].details["n_gaps"] == 1
        assert gap_alerts[0].details["max_gap_seconds"] > 300_000

    def test_sorted_no_datetime_alerts(self):
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1, 0, 0)
        df = pl.DataFrame({"ts": [base + timedelta(hours=i) for i in range(100)]})
        alerts = _get_alerts(df)
        datetime_alerts = [
            a
            for a in alerts
            if a.alert_type
            in (AlertType.UNSORTED_DATES, AlertType.IRREGULAR_INTERVALS, AlertType.LARGE_GAPS)
        ]
        assert len(datetime_alerts) == 0

    def test_single_date_no_datetime_alerts(self):
        from datetime import date

        df = pl.DataFrame({"d": [date(2024, 6, 15)]})
        alerts = _get_alerts(df)
        datetime_alerts = [
            a
            for a in alerts
            if a.alert_type
            in (AlertType.UNSORTED_DATES, AlertType.IRREGULAR_INTERVALS, AlertType.LARGE_GAPS)
        ]
        assert len(datetime_alerts) == 0


class TestCleanData:
    def test_no_alerts_on_clean_data(self):
        df = pl.DataFrame(
            {
                "score": [10, 20, 30, 40, 50, 10, 20, 30, 40, 50],
                "city": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
                "active": [True, False, True, False, True, False, True, False, True, False],
            }
        )
        alerts = _get_alerts(df)
        col_alerts = [
            a for a in alerts if a.column is not None and a.alert_type != AlertType.UNIFORM
        ]
        assert len(col_alerts) == 0


class TestTimeSeriesAlerts:
    def test_linear_trend_non_stationary_alert(self):
        df = pl.DataFrame({"val": [float(i) for i in range(200)]})
        alerts = _get_alerts(df)
        col_alerts = _alerts_for_column(alerts, "val")
        assert any(a.alert_type == AlertType.NON_STATIONARY for a in col_alerts)

    def test_white_noise_no_non_stationary_alert(self):
        rng = random.Random(7)
        df = pl.DataFrame({"val": [rng.gauss(0, 1) for _ in range(500)]})
        alerts = _get_alerts(df)
        col_alerts = _alerts_for_column(alerts, "val")
        assert not any(a.alert_type == AlertType.NON_STATIONARY for a in col_alerts)

    def test_random_no_non_stationary_alert(self):
        rng = random.Random(42)
        df = pl.DataFrame({"val": [rng.random() for _ in range(200)]})
        alerts = _get_alerts(df)
        col_alerts = _alerts_for_column(alerts, "val")
        assert not any(a.alert_type == AlertType.NON_STATIONARY for a in col_alerts)

    def test_ts_active_false_no_non_stationary_alert(self):
        df = pl.DataFrame({"val": [float(i) for i in range(200)]})
        alerts = _get_alerts(df, ProfileConfig(ts_active=False))
        col_alerts = _alerts_for_column(alerts, "val")
        assert not any(a.alert_type == AlertType.NON_STATIONARY for a in col_alerts)
