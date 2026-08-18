from __future__ import annotations

from datetime import date, datetime, time, timedelta

import polars as pl
import pytest

from dataxid_profiling._analyzers._datetime import analyze_datetime
from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._type_inference import ColumnType


@pytest.fixture
def config() -> ProfileConfig:
    return ProfileConfig()


class TestDatetimeBasicStats:
    def test_count_and_missing(self, datetime_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_datetime(datetime_df, "created_at", config)
        assert stats.count == 6
        assert stats.missing_count == 1
        assert stats.missing_pct == pytest.approx(1 / 6)

    def test_column_metadata(self, datetime_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_datetime(datetime_df, "created_at", config)
        assert stats.column_name == "created_at"
        assert stats.column_type == ColumnType.DATETIME

    def test_distinct(self, datetime_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_datetime(datetime_df, "created_at", config)
        assert stats.distinct_count == 5
        assert stats.distinct_pct == pytest.approx(5 / 6)


class TestDatetimeMinMaxRange:
    def test_datetime_min_max(self, config: ProfileConfig):
        df = pl.DataFrame(
            {
                "ts": [datetime(2024, 1, 1), datetime(2024, 6, 15), datetime(2024, 12, 31)],
            }
        )
        stats = analyze_datetime(df, "ts", config)
        assert stats.min is not None
        assert stats.max is not None
        assert "2024-01-01" in stats.min
        assert "2024-12-31" in stats.max

    def test_date_min_max(self, config: ProfileConfig):
        df = pl.DataFrame(
            {
                "d": [date(2023, 1, 1), date(2023, 6, 15), date(2023, 12, 31)],
            }
        )
        stats = analyze_datetime(df, "d", config)
        assert "2023-01-01" in stats.min
        assert "2023-12-31" in stats.max

    def test_range_not_none(self, datetime_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_datetime(datetime_df, "created_at", config)
        assert stats.range is not None

    def test_date_range_value(self, config: ProfileConfig):
        df = pl.DataFrame(
            {
                "d": [date(2024, 1, 1), date(2024, 1, 11)],
            }
        )
        stats = analyze_datetime(df, "d", config)
        assert stats.range is not None
        assert "10" in stats.range


class TestDatetimeTimeAndDuration:
    def test_time_column(self, config: ProfileConfig):
        df = pl.DataFrame({"t": [time(8, 0), time(12, 0), time(18, 0)]})
        stats = analyze_datetime(df, "t", config)
        assert stats.count == 3
        assert stats.min is not None
        assert stats.max is not None

    def test_duration_column(self, config: ProfileConfig):
        df = pl.DataFrame({"dur": [timedelta(hours=1), timedelta(hours=5), timedelta(hours=10)]})
        stats = analyze_datetime(df, "dur", config)
        assert stats.count == 3
        assert stats.min is not None


class TestDatetimeEdgeCases:
    def test_all_null(self, config: ProfileConfig):
        df = pl.DataFrame({"ts": pl.Series([None, None], dtype=pl.Datetime)})
        stats = analyze_datetime(df, "ts", config)
        assert stats.count == 2
        assert stats.missing_count == 2
        assert stats.min is None
        assert stats.max is None
        assert stats.range is None

    def test_empty_dataframe(self, config: ProfileConfig):
        df = pl.DataFrame({"ts": pl.Series([], dtype=pl.Datetime)})
        stats = analyze_datetime(df, "ts", config)
        assert stats.count == 0

    def test_single_value(self, config: ProfileConfig):
        df = pl.DataFrame({"d": [date(2024, 6, 15)]})
        stats = analyze_datetime(df, "d", config)
        assert stats.distinct_count == 1
        assert stats.min == stats.max

    def test_single_value_time_series(self, config: ProfileConfig):
        """Single row: is_sorted=True, no gaps, no autocorrelation."""
        df = pl.DataFrame({"ts": [datetime(2024, 6, 15, 12, 0)]})
        stats = analyze_datetime(df, "ts", config)
        assert stats.is_sorted is True
        assert stats.is_monotonic_increasing is True
        assert stats.n_gaps == 0
        assert stats.autocorrelation_lag1 is None

    def test_all_same_date(self, config: ProfileConfig):
        """All dates identical: is_sorted=True, no gaps, interval 0."""
        df = pl.DataFrame({"d": [date(2024, 6, 15)] * 5})
        stats = analyze_datetime(df, "d", config)
        assert stats.is_sorted is True
        assert stats.is_monotonic_increasing is True
        assert stats.n_gaps == 0
        assert stats.sampling_interval_median_seconds == 0.0

    def test_date_type_time_series(self, config: ProfileConfig):
        """Date type should work with .dt.total_microseconds()."""
        df = pl.DataFrame(
            {
                "d": [
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 4),
                    date(2024, 1, 5),
                ]
            }
        )
        stats = analyze_datetime(df, "d", config)
        assert stats.is_sorted is True
        assert stats.is_regular_interval is True
        assert stats.sampling_interval_median_seconds == 86400.0
        assert stats.n_gaps == 0

    def test_null_heavy(self, config: ProfileConfig):
        """Nulls are dropped, remaining values sorted correctly."""
        df = pl.DataFrame(
            {
                "ts": [
                    datetime(2024, 1, 1, 0, 0),
                    None,
                    datetime(2024, 1, 1, 2, 0),
                    None,
                    datetime(2024, 1, 1, 4, 0),
                ]
            }
        )
        stats = analyze_datetime(df, "ts", config)
        assert stats.count == 5
        assert stats.missing_count == 2
        assert stats.is_sorted is True
        assert stats.sampling_interval_median_seconds == 7200.0


class TestDatetimeTimeSeries:
    def test_sorted_hourly(self, config: ProfileConfig):
        """100 rows, hourly — perfectly sorted and regular."""
        base = datetime(2024, 1, 1, 0, 0)
        df = pl.DataFrame({"ts": [base + timedelta(hours=i) for i in range(100)]})
        stats = analyze_datetime(df, "ts", config)
        assert stats.is_sorted is True
        assert stats.is_monotonic_increasing is True
        assert stats.is_monotonic_decreasing is False
        assert stats.is_regular_interval is True
        assert stats.sampling_interval_median_seconds == 3600.0
        assert stats.sampling_interval_mean_seconds == 3600.0
        assert stats.n_gaps == 0
        assert stats.max_gap_seconds is None

    def test_sorted_descending(self, config: ProfileConfig):
        """Descending order: is_sorted=True, decreasing."""
        base = datetime(2024, 12, 31, 23, 0)
        df = pl.DataFrame({"ts": [base - timedelta(hours=i) for i in range(10)]})
        stats = analyze_datetime(df, "ts", config)
        assert stats.is_sorted is True
        assert stats.is_monotonic_increasing is False
        assert stats.is_monotonic_decreasing is True

    def test_unsorted(self, config: ProfileConfig):
        """Random order: is_sorted=False."""
        df = pl.DataFrame(
            {
                "ts": [
                    datetime(2024, 1, 3),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                ]
            }
        )
        stats = analyze_datetime(df, "ts", config)
        assert stats.is_sorted is False
        assert stats.is_monotonic_increasing is False
        assert stats.is_monotonic_decreasing is False

    def test_gap_detection(self, config: ProfileConfig):
        """Large gap > 2x median interval: n_gaps=1."""
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
        stats = analyze_datetime(df, "ts", config)
        assert stats.is_sorted is True
        assert stats.n_gaps == 1
        assert stats.max_gap_seconds is not None
        assert stats.max_gap_seconds > 300_000  # ~94 hours in seconds

    def test_no_gap_below_multiplier(self, config: ProfileConfig):
        """Small variations within multiplier: no gaps."""
        df = pl.DataFrame(
            {
                "ts": [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 2, 0),
                    datetime(2024, 1, 1, 3, 0),
                    datetime(2024, 1, 1, 6, 0),
                ]
            }
        )
        stats = analyze_datetime(df, "ts", config)
        # Median = (7200+3600+10800)/3 ~ 7200, 2x = 14400
        # max diff = 10800 < 14400, so no gaps
        assert stats.n_gaps == 0

    def test_irregular_interval(self, config: ProfileConfig):
        """High std/mean ratio: is_regular_interval=False."""
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
        stats = analyze_datetime(df, "ts", config)
        assert stats.is_sorted is True
        assert stats.is_regular_interval is False
        assert stats.sampling_interval_std_seconds is not None
        assert stats.sampling_interval_mean_seconds is not None
        assert stats.sampling_interval_std_seconds / stats.sampling_interval_mean_seconds > 0.01

    def test_autocorrelation_linear_trend(self, config: ProfileConfig):
        """Hourly data: strong positive lag-1 autocorrelation."""
        base = datetime(2024, 1, 1, 0, 0)
        df = pl.DataFrame({"ts": [base + timedelta(hours=i) for i in range(100)]})
        stats = analyze_datetime(df, "ts", config)
        assert stats.autocorrelation_lag1 is not None
        assert stats.autocorrelation_lag1 > 0.9

    def test_autocorrelation_random(self, config: ProfileConfig):
        """Random timestamps: autocorrelation near zero."""
        from random import randrange

        base = datetime(2024, 1, 1)
        df = pl.DataFrame({"ts": [base + timedelta(days=randrange(0, 365)) for _ in range(50)]})
        stats = analyze_datetime(df, "ts", config)
        assert stats.autocorrelation_lag1 is not None
        assert abs(stats.autocorrelation_lag1) < 0.5

    def test_two_rows_no_autocorr(self, config: ProfileConfig):
        """2 rows: n-1 < 2 → no autocorrelation."""
        df = pl.DataFrame(
            {
                "ts": [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 1, 0),
                ]
            }
        )
        stats = analyze_datetime(df, "ts", config)
        assert stats.autocorrelation_lag1 is None

    def test_five_rows_mixed_interval(self, config: ProfileConfig):
        """Sanity: 5 rows daily, median and mean should be 86400."""
        df = pl.DataFrame(
            {
                "d": [
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 4),
                    date(2024, 1, 5),
                ]
            }
        )
        stats = analyze_datetime(df, "d", config)
        assert stats.sampling_interval_median_seconds == 86400.0
        assert stats.sampling_interval_mean_seconds == 86400.0


class TestDatetimeGapStats:
    def test_gap_stats_present(self, config: ProfileConfig):
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
        stats = analyze_datetime(df, "ts", config)
        assert stats.n_gaps == 1
        assert stats.gap_min_seconds is not None
        assert stats.gap_mean_seconds is not None
        assert stats.gap_std_seconds == 0.0
        assert stats.gap_threshold_seconds is not None
        assert stats.gap_threshold_seconds > 0

    def test_gap_stats_none_when_no_gap(self, config: ProfileConfig):
        df = pl.DataFrame(
            {"ts": [datetime(2024, 1, 1, 0, 0) + timedelta(hours=i) for i in range(10)]}
        )
        stats = analyze_datetime(df, "ts", config)
        assert stats.n_gaps == 0
        assert stats.gap_min_seconds is None
        assert stats.gap_mean_seconds is None
        assert stats.gap_std_seconds is None

    def test_gap_indices_match_gaps(self, config: ProfileConfig):
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
        stats = analyze_datetime(df, "ts", config)
        assert stats.gap_indices == [2]
        assert len(stats.interval_values) == 5

    def test_gap_plot_samples_large_series(self):
        df = pl.DataFrame(
            {"ts": [datetime(2024, 1, 1, 0, 0) + timedelta(hours=i) for i in range(20_000)]}
        )
        config = ProfileConfig(ts_line_max_points=100)
        stats = analyze_datetime(df, "ts", config)
        assert len(stats.interval_values) == 100

    def test_gap_threshold_respects_multiplier(self):
        base = datetime(2024, 1, 1, 0, 0)
        df = pl.DataFrame({"ts": [base + timedelta(hours=i) for i in range(10)]})
        config = ProfileConfig(ts_gap_multiplier=5.0)
        stats = analyze_datetime(df, "ts", config)
        assert stats.gap_threshold_seconds == 3600.0 * 5.0
