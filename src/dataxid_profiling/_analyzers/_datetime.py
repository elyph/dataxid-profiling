from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from dataxid_profiling._analyzers import DatetimeStats
from dataxid_profiling._type_inference import ColumnType

if TYPE_CHECKING:
    from dataxid_profiling._config import ProfileConfig


def analyze_datetime(df: pl.DataFrame, col_name: str, config: ProfileConfig) -> DatetimeStats:
    col = pl.col(col_name)
    n_rows = df.height

    if n_rows == 0:
        return _empty_stats(col_name)

    row = df.select(
        col.null_count().alias("missing_count"),
        col.drop_nulls().n_unique().alias("distinct_count"),
        col.min().alias("min"),
        col.max().alias("max"),
    ).row(0, named=True)

    missing_count: int = row["missing_count"]
    distinct_count: int = row["distinct_count"]
    min_val = row["min"]
    max_val = row["max"]

    range_str = _compute_range(min_val, max_val)

    # --- Time series analysis ---
    ts = _analyze_timeseries(df, col_name, config)

    return DatetimeStats(
        column_name=col_name,
        column_type=ColumnType.DATETIME,
        count=n_rows,
        missing_count=missing_count,
        missing_pct=missing_count / n_rows,
        distinct_count=distinct_count,
        distinct_pct=distinct_count / n_rows,
        min=str(min_val) if min_val is not None else None,
        max=str(max_val) if max_val is not None else None,
        range=range_str,
        **ts,
    )


def _analyze_timeseries(df: pl.DataFrame, col_name: str, config: ProfileConfig) -> dict[str, Any]:
    """Compute time series metrics: sorting, sampling interval, gaps, autocorrelation."""
    non_null = df.select(pl.col(col_name).drop_nulls())
    n = non_null.height

    if n < 2:
        return {
            "is_sorted": True,
            "is_monotonic_increasing": True,
            "is_monotonic_decreasing": False,
        }

    # Check if sorted (monotonic)
    diffs_ms = non_null.select(
        pl.col(col_name).diff().dt.total_microseconds().alias("diff_ms")
    ).drop_nulls()

    diffs_values = diffs_ms.get_column("diff_ms")
    is_increasing = bool((diffs_values >= 0).all())
    is_decreasing = bool((diffs_values <= 0).all())
    is_sorted = is_increasing or is_decreasing

    if diffs_values.len() == 0:
        return {
            "is_sorted": is_sorted,
            "is_monotonic_increasing": is_increasing,
            "is_monotonic_decreasing": is_decreasing,
        }

    # Sampling intervals (in seconds), absolute
    intervals = diffs_values.abs().cast(pl.Float64) / 1_000_000

    interval_median = intervals.median()
    interval_mean = intervals.mean()
    interval_std = intervals.std()

    # Regular interval check: std < 1% of mean
    is_regular = False
    if interval_mean is not None and interval_std is not None and interval_mean > 0:
        is_regular = (interval_std / interval_mean) < 0.01

    # Gap detection: diff > multiplier × median
    n_gaps = 0
    max_gap = None
    if interval_median is not None and interval_median > 0:
        gap_threshold = interval_median * config.ts_gap_multiplier
        gaps = intervals.filter(intervals > gap_threshold)
        n_gaps = gaps.len()
        if n_gaps > 0:
            max_gap = gaps.max()

    # Lag-1 autocorrelation
    acf_lag1 = None
    if n > 2:
        try:
            vals = non_null.get_column(col_name).cast(pl.Float64)
            mean_val = vals.mean()
            var_val = vals.var()
            if var_val is not None and var_val > 0:
                shifted = vals.slice(1, n - 1)
                original = vals.slice(0, n - 1)
                acf_lag1 = float(
                    ((original - mean_val) * (shifted - mean_val)).sum()
                    / ((n - 1) * var_val)
                )
        except Exception:
            pass

    return {
        "is_sorted": is_sorted,
        "is_monotonic_increasing": is_increasing,
        "is_monotonic_decreasing": is_decreasing,
        "sampling_interval_median_seconds": round(interval_median, 3) if interval_median is not None else None,
        "sampling_interval_mean_seconds": round(interval_mean, 3) if interval_mean is not None else None,
        "sampling_interval_std_seconds": round(interval_std, 3) if interval_std is not None else None,
        "is_regular_interval": is_regular,
        "n_gaps": n_gaps,
        "max_gap_seconds": round(max_gap, 3) if max_gap is not None else None,
        "autocorrelation_lag1": round(acf_lag1, 6) if acf_lag1 is not None else None,
    }


def _compute_range(min_val: Any, max_val: Any) -> str | None:
    if min_val is None or max_val is None:
        return None
    try:
        delta = max_val - min_val
        return str(delta)
    except TypeError:
        return None


def _empty_stats(col_name: str) -> DatetimeStats:
    return DatetimeStats(
        column_name=col_name,
        column_type=ColumnType.DATETIME,
        count=0,
        missing_count=0,
        missing_pct=0.0,
    )
