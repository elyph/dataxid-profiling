from __future__ import annotations

import math
import random

import polars as pl
import pytest

from dataxid_profiling._analyzers._numeric import analyze_numeric
from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._type_inference import ColumnType


@pytest.fixture
def config() -> ProfileConfig:
    return ProfileConfig()


class TestNumericBasicStats:
    def test_count_and_missing(self, numeric_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_numeric(numeric_df, "age", config)
        assert stats.count == 6
        assert stats.missing_count == 1
        assert stats.missing_pct == pytest.approx(1 / 6)

    def test_column_metadata(self, numeric_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_numeric(numeric_df, "age", config)
        assert stats.column_name == "age"
        assert stats.column_type == ColumnType.NUMERIC

    def test_distinct(self, numeric_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_numeric(numeric_df, "age", config)
        assert stats.distinct_count >= 1
        assert 0.0 <= stats.distinct_pct <= 1.0


class TestNumericDescriptiveStats:
    def test_mean(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.mean == pytest.approx(30.0)

    def test_std(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.std is not None
        assert stats.std > 0

    def test_min_max_range(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [5, 10, 15, 20, 25]})
        stats = analyze_numeric(df, "val", config)
        assert stats.min == pytest.approx(5.0)
        assert stats.max == pytest.approx(25.0)
        assert stats.range == pytest.approx(20.0)

    def test_quantiles(self, config: ProfileConfig):
        df = pl.DataFrame({"val": list(range(1, 101))})
        stats = analyze_numeric(df, "val", config)
        assert stats.q25 is not None
        assert stats.median is not None
        assert stats.q75 is not None
        assert stats.q25 < stats.median < stats.q75

    def test_iqr(self, config: ProfileConfig):
        df = pl.DataFrame({"val": list(range(1, 101))})
        stats = analyze_numeric(df, "val", config)
        assert stats.iqr is not None
        assert stats.iqr == pytest.approx(stats.q75 - stats.q25)

    def test_skewness_symmetric(self, config: ProfileConfig):
        df = pl.DataFrame({"val": list(range(1, 101))})
        stats = analyze_numeric(df, "val", config)
        assert stats.skewness is not None
        assert abs(stats.skewness) < 0.5

    def test_kurtosis(self, config: ProfileConfig):
        df = pl.DataFrame({"val": list(range(1, 101))})
        stats = analyze_numeric(df, "val", config)
        assert stats.kurtosis is not None


class TestNumericZerosAndNegatives:
    def test_zeros(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [0, 0, 1, 2, 3]})
        stats = analyze_numeric(df, "val", config)
        assert stats.zeros_count == 2
        assert stats.zeros_pct == pytest.approx(0.4)

    def test_negatives(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [-5, -3, 0, 2, 4]})
        stats = analyze_numeric(df, "val", config)
        assert stats.negative_count == 2
        assert stats.negative_pct == pytest.approx(0.4)

    def test_no_zeros_no_negatives(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1, 2, 3]})
        stats = analyze_numeric(df, "val", config)
        assert stats.zeros_count == 0
        assert stats.negative_count == 0


class TestNumericHistogram:
    def test_histogram_not_empty(self, numeric_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_numeric(numeric_df, "salary", config)
        assert len(stats.histogram) > 0

    def test_histogram_has_breakpoint_and_count(self, config: ProfileConfig):
        df = pl.DataFrame({"val": list(range(100))})
        stats = analyze_numeric(df, "val", config)
        for bucket in stats.histogram:
            assert "breakpoint" in bucket
            assert "count" in bucket

    def test_histogram_custom_bins(self):
        df = pl.DataFrame({"val": list(range(100))})
        config = ProfileConfig(histogram_bins=10)
        stats = analyze_numeric(df, "val", config)
        assert len(stats.histogram) == 10


class TestNumericEdgeCases:
    def test_all_null(self, config: ProfileConfig):
        df = pl.DataFrame({"val": pl.Series([None, None, None], dtype=pl.Float64)})
        stats = analyze_numeric(df, "val", config)
        assert stats.count == 3
        assert stats.missing_count == 3
        assert stats.mean is None

    def test_single_value(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [42]})
        stats = analyze_numeric(df, "val", config)
        assert stats.mean == pytest.approx(42.0)
        assert stats.min == pytest.approx(42.0)
        assert stats.max == pytest.approx(42.0)
        assert stats.range == pytest.approx(0.0)

    def test_empty_dataframe(self, config: ProfileConfig):
        df = pl.DataFrame({"val": pl.Series([], dtype=pl.Int64)})
        stats = analyze_numeric(df, "val", config)
        assert stats.count == 0
        assert stats.mean is None

    def test_float_types(self, config: ProfileConfig):
        df = pl.DataFrame({"val": pl.Series([1.5, 2.5, 3.5], dtype=pl.Float32)})
        stats = analyze_numeric(df, "val", config)
        assert stats.mean == pytest.approx(2.5, abs=0.01)


class TestNumericNewMetrics:
    def test_sum(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [10, 20, 30]})
        stats = analyze_numeric(df, "val", config)
        assert stats.sum == pytest.approx(60.0)

    def test_variance(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.variance is not None
        assert stats.variance > 0
        assert stats.std is not None
        assert stats.variance == pytest.approx(stats.std**2, rel=1e-3)

    def test_cv(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.cv is not None
        assert stats.cv == pytest.approx(stats.std / abs(stats.mean), rel=1e-3)

    def test_cv_zero_mean(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [-1.0, 0.0, 1.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.cv is None

    def test_mad(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.mad is not None
        assert stats.mad == pytest.approx(1.0)

    def test_p5_p95(self, config: ProfileConfig):
        df = pl.DataFrame({"val": list(range(1, 101))})
        stats = analyze_numeric(df, "val", config)
        assert stats.p5 is not None
        assert stats.p95 is not None
        assert stats.p5 < stats.q25
        assert stats.p95 > stats.q75

    def test_n_infinite(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1.0, float("inf"), -float("inf"), 4.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.n_infinite == 2

    def test_n_infinite_none_for_int(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        stats = analyze_numeric(df, "val", config)
        assert stats.n_infinite == 0

    def test_monotonic_increase(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        stats = analyze_numeric(df, "val", config)
        assert stats.monotonic_increase is True
        assert stats.monotonic_decrease is False

    def test_monotonic_decrease(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [5, 4, 3, 2, 1]})
        stats = analyze_numeric(df, "val", config)
        assert stats.monotonic_increase is False
        assert stats.monotonic_decrease is True

    def test_monotonic_constant(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [3, 3, 3, 3]})
        stats = analyze_numeric(df, "val", config)
        assert stats.monotonic_increase is True
        assert stats.monotonic_decrease is True

    def test_monotonic_non_monotonic(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1, 3, 2, 4]})
        stats = analyze_numeric(df, "val", config)
        assert stats.monotonic_increase is False
        assert stats.monotonic_decrease is False

    def test_monotonic_with_nulls(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1, None, 3, None, 5]})
        stats = analyze_numeric(df, "val", config)
        assert stats.monotonic_increase is True

    def test_value_counts(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1, 1, 1, 2, 2, 3]})
        stats = analyze_numeric(df, "val", config)
        assert len(stats.value_counts) > 0
        assert stats.value_counts[0]["value"] == 1
        assert stats.value_counts[0]["count"] == 3

    def test_value_counts_respects_n_top(self):
        df = pl.DataFrame({"val": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]})
        config = ProfileConfig(n_top_values=3)
        stats = analyze_numeric(df, "val", config)
        assert len(stats.value_counts) == 3


class TestNumericTimeSeries:
    def test_linear_trend_is_timeseries(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [float(i) for i in range(200)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries is True

    def test_random_is_not_timeseries(self, config: ProfileConfig):
        rng = random.Random(42)
        df = pl.DataFrame({"val": [rng.random() for _ in range(200)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries is False

    def test_constant_is_not_timeseries(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [3.0] * 100})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries is False
        assert stats.adf_pvalue is None

    def test_seasonal_sine_is_timeseries(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [math.sin(2 * math.pi * i / 7) for i in range(100)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries is True

    def test_ts_active_false_disables_detection(self):
        df = pl.DataFrame({"val": [float(i) for i in range(200)]})
        config = ProfileConfig(ts_active=False)
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries is False
        assert stats.adf_pvalue is None

    def test_short_series_no_timeseries(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1.0, 2.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries is False
        assert stats.adf_pvalue is None

    def test_low_cardinality_not_timeseries(self, config: ProfileConfig):
        """Columns with <6 distinct values are not meaningful time series.

        Mirrors ydata's exclusion of binary/ordinal columns from the numeric
        TS list. A monotonic 0/1 sequence (e.g. yr) would otherwise pass the
        autocorrelation threshold because of its repeating pattern.
        """
        df = pl.DataFrame(
            {"yr": [0] * 90 + [1] * 90, "season": [i % 4 + 1 for i in range(180)]}
        )
        for col in ("yr", "season"):
            stats = analyze_numeric(df, col, config)
            assert stats.is_timeseries is False, col
            assert stats.adf_pvalue is None, col
            assert stats.is_seasonal is False, col

    def test_mid_cardinality_stays_timeseries(self, config: ProfileConfig):
        """mnth (12) and weekday (7) have enough distinct values to be TS."""
        df = pl.DataFrame(
            {"mnth": [i % 12 + 1 for i in range(200)], "weekday": [i % 7 for i in range(200)]}
        )
        for col in ("mnth", "weekday"):
            stats = analyze_numeric(df, col, config)
            assert stats.is_timeseries is True, col

    def test_null_heavy_does_not_crash(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [float(i) if i % 3 != 0 else None for i in range(60)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries in (True, False)

    def test_linear_trend_non_stationary(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [float(i) for i in range(200)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries is True
        assert stats.adf_pvalue is not None
        assert stats.is_stationary is False
        assert stats.adf_statistic is not None

    def test_adf_statistic_none_for_non_ts(self, config: ProfileConfig):
        rng = random.Random(42)
        df = pl.DataFrame({"val": [rng.random() for _ in range(200)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries is False
        assert stats.adf_statistic is None
        assert stats.adf_pvalue is None

    def test_white_noise_stationary(self, config: ProfileConfig):
        rng = random.Random(7)
        df = pl.DataFrame({"val": [rng.gauss(0, 1) for _ in range(500)]})
        stats = analyze_numeric(df, "val", config)
        if stats.is_timeseries:
            assert stats.adf_pvalue is not None
            assert stats.is_stationary is True

    def test_seasonal_effective_stationary_false(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [math.sin(2 * math.pi * i / 7) for i in range(200)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries is True
        assert stats.is_seasonal is True
        assert stats.is_stationary is True
        assert stats.is_effective_stationary is False

    def test_adf_sampling_caps_points(self):
        df = pl.DataFrame({"val": [float(i) for i in range(50_000)]})
        config = ProfileConfig(ts_adf_max_points=1_000)
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries is True
        assert stats.adf_pvalue is not None
        assert 0.0 <= stats.adf_pvalue <= 1.0

    def test_line_data_caps_points(self):
        df = pl.DataFrame({"val": [float(i) for i in range(50_000)]})
        config = ProfileConfig(ts_line_max_points=100)
        stats = analyze_numeric(df, "val", config)
        assert len(stats.line_data) == 100

    def test_line_data_empty_for_non_ts(self, config: ProfileConfig):
        rng = random.Random(42)
        df = pl.DataFrame({"val": [rng.random() for _ in range(200)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.line_data == []

    def test_acf_pacf_values_present_for_ts(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [float(i) for i in range(200)]})
        stats = analyze_numeric(df, "val", config)
        assert len(stats.acf_values) > 0
        assert len(stats.pacf_values) > 0
        assert len(stats.acf_values) == len(stats.pacf_values)

    def test_acf_pacf_empty_for_short_series(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1.0, 2.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.acf_values == []
        assert stats.pacf_values == []

    def test_acf_pacf_sampling_caps_points(self):
        df = pl.DataFrame({"val": [float(i) for i in range(50_000)]})
        config = ProfileConfig(ts_acf_pacf_max_points=1_000)
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries is True
        assert len(stats.acf_values) > 0
        assert len(stats.pacf_values) > 0


class TestNumericSeasonality:
    def test_sine_is_seasonal(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [math.sin(2 * math.pi * i / 7) for i in range(200)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_timeseries is True
        assert stats.is_seasonal is True
        assert len(stats.seasonal_periods) > 0
        assert any(p == pytest.approx(7, abs=0.5) for p in stats.seasonal_periods)

    def test_white_noise_not_seasonal(self, config: ProfileConfig):
        rng = random.Random(7)
        df = pl.DataFrame({"val": [rng.gauss(0, 1) for _ in range(500)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_seasonal is False
        assert stats.seasonal_periods == []

    def test_linear_trend_not_seasonal(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [float(i) for i in range(200)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_seasonal is False
        assert stats.seasonal_periods == []

    def test_short_series_not_seasonal(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [math.sin(2 * math.pi * i / 7) for i in range(15)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_seasonal is False
        assert stats.seasonal_periods == []

    def test_ts_active_false_not_seasonal(self):
        df = pl.DataFrame({"val": [math.sin(2 * math.pi * i / 7) for i in range(200)]})
        config = ProfileConfig(ts_active=False)
        stats = analyze_numeric(df, "val", config)
        assert stats.is_seasonal is False
        assert stats.seasonal_periods == []

    def test_low_cardinality_skip_seasonality(self, config: ProfileConfig):
        """Columns with <6 distinct values (e.g. season with 4 values) cannot
        carry a meaningful Fourier period — FFT would report spurious peaks.
        ydata excludes these from the numeric TS list; we short-circuit."""
        # 4 distinct values, 200 rows, alternating 1,2,3,4,1,2,3,4,...
        df = pl.DataFrame({"val": [i % 4 + 1 for i in range(200)]})
        stats = analyze_numeric(df, "val", config)
        assert stats.is_seasonal is False
        assert stats.seasonal_periods == []

    def test_seasonal_periods_non_empty_when_seasonal(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [math.sin(2 * math.pi * i / 7) for i in range(200)]})
        stats = analyze_numeric(df, "val", config)
        if stats.is_seasonal:
            assert len(stats.seasonal_periods) > 0

    def test_harmonics_not_in_periods(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [math.sin(2 * math.pi * i / 7) for i in range(200)]})
        stats = analyze_numeric(df, "val", config)
        if stats.is_seasonal:
            fundamental = min(stats.seasonal_periods)
            for p in stats.seasonal_periods:
                if p > fundamental:
                    ratio = p / fundamental
                    assert abs(ratio - round(ratio)) >= 0.01


class TestNumericSortby:
    def test_sortby_missing_column_falls_back(self):
        df = pl.DataFrame({"val": [float(i) for i in range(200)]})
        cfg = ProfileConfig(ts_sortby="does_not_exist")
        stats = analyze_numeric(df, "val", cfg)
        assert stats.is_timeseries is True

    def test_sortby_numeric_index_recovers_timeseries(self):
        vals = [math.sin(2 * math.pi * i / 7) for i in range(200)]
        shuffled = vals[::2] + vals[1::2]
        df = pl.DataFrame({"val": shuffled, "idx": list(range(200))})
        cfg = ProfileConfig(ts_sortby="idx")
        stats = analyze_numeric(df, "val", cfg)
        assert stats.is_timeseries is True
