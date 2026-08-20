from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from dataxid_profiling._analyzers import NumericStats
from dataxid_profiling._type_inference import ColumnType

if TYPE_CHECKING:
    from dataxid_profiling._config import ProfileConfig


def analyze_numeric(df: pl.DataFrame, col_name: str, config: ProfileConfig) -> NumericStats:
    col = pl.col(col_name)
    n_rows = df.height

    if n_rows == 0:
        return _empty_stats(col_name)

    row = df.select(
        col.count().alias("count"),
        col.null_count().alias("missing_count"),
        col.n_unique().alias("distinct_count"),
        col.mean().alias("mean"),
        col.std().alias("std"),
        col.var().alias("variance"),
        col.sum().alias("sum"),
        col.min().alias("min"),
        col.max().alias("max"),
        col.quantile(0.05, interpolation="linear").alias("p5"),
        col.quantile(0.25, interpolation="linear").alias("q25"),
        col.median().alias("median"),
        col.quantile(0.75, interpolation="linear").alias("q75"),
        col.quantile(0.95, interpolation="linear").alias("p95"),
        col.skew().alias("skewness"),
        col.kurtosis().alias("kurtosis"),
        col.eq(0).sum().alias("zeros_count"),
        col.lt(0).sum().alias("negative_count"),
        col.is_infinite().sum().alias("n_infinite"),
        (col - col.median()).abs().median().alias("mad"),
    ).row(0, named=True)

    missing_count: int = row["missing_count"]
    distinct_count: int = row["distinct_count"]
    min_val = row["min"]
    max_val = row["max"]
    q25 = row["q25"]
    q75 = row["q75"]
    mean_val = _safe_float(row["mean"])
    std_val = _safe_float(row["std"])

    range_val = (max_val - min_val) if min_val is not None and max_val is not None else None
    iqr = (q75 - q25) if q25 is not None and q75 is not None else None
    cv = None
    if std_val is not None and mean_val and mean_val != 0.0:
        cv = std_val / abs(mean_val)

    monotonic_inc, monotonic_dec = _check_monotonic(df, col_name)
    histogram = _compute_histogram(df, col_name, config.histogram_bins)
    top_values = _compute_value_counts(df, col_name, config.n_top_values)

    is_ts = _detect_timeseries(df, col_name, config)
    adf_statistic, adf_pvalue, is_stationary = (
        _adf_stationarity(df, col_name, config) if is_ts else (None, None, False)
    )
    line_data = _extract_line_data(df, col_name, config) if is_ts else []
    acf_values, pacf_values = _compute_acf_pacf(df, col_name, config) if is_ts else ([], [])
    is_seasonal, seasonal_periods = (
        _detect_seasonality(df, col_name, config) if is_ts else (False, [])
    )
    is_effective_stationary = is_stationary and not is_seasonal

    return NumericStats(
        column_name=col_name,
        column_type=ColumnType.NUMERIC,
        count=n_rows,
        missing_count=missing_count,
        missing_pct=missing_count / n_rows if n_rows > 0 else 0.0,
        distinct_count=distinct_count,
        distinct_pct=distinct_count / n_rows if n_rows > 0 else 0.0,
        mean=mean_val,
        std=std_val,
        variance=_safe_float(row["variance"]),
        sum=_safe_float(row["sum"]),
        min=_safe_float(min_val),
        max=_safe_float(max_val),
        range=_safe_float(range_val),
        q25=_safe_float(q25),
        median=_safe_float(row["median"]),
        q75=_safe_float(q75),
        p5=_safe_float(row["p5"]),
        p95=_safe_float(row["p95"]),
        iqr=_safe_float(iqr),
        cv=_safe_float(cv),
        mad=_safe_float(row["mad"]),
        skewness=_safe_float(row["skewness"]),
        kurtosis=_safe_float(row["kurtosis"]),
        zeros_count=row["zeros_count"],
        zeros_pct=row["zeros_count"] / n_rows if n_rows > 0 else 0.0,
        negative_count=row["negative_count"],
        negative_pct=row["negative_count"] / n_rows if n_rows > 0 else 0.0,
        n_infinite=row["n_infinite"],
        monotonic_increase=monotonic_inc,
        monotonic_decrease=monotonic_dec,
        histogram=histogram,
        value_counts=top_values,
        is_timeseries=is_ts,
        adf_statistic=adf_statistic,
        adf_pvalue=adf_pvalue,
        is_stationary=is_stationary,
        is_effective_stationary=is_effective_stationary,
        line_data=line_data,
        acf_values=acf_values,
        pacf_values=pacf_values,
        is_seasonal=is_seasonal,
        seasonal_periods=seasonal_periods,
    )


def _check_monotonic(df: pl.DataFrame, col_name: str) -> tuple[bool, bool]:
    """Check if column is monotonically increasing or decreasing (ignoring nulls)."""
    non_null = df.filter(pl.col(col_name).is_not_null())
    if non_null.height < 2:
        return False, False

    diffs = non_null.select(pl.col(col_name).diff().drop_nulls().alias("d"))
    if diffs.height == 0:
        return False, False

    row = diffs.select(
        (pl.col("d") >= 0).all().alias("inc"),
        (pl.col("d") <= 0).all().alias("dec"),
    ).row(0, named=True)

    return bool(row["inc"]), bool(row["dec"])


def _compute_value_counts(df: pl.DataFrame, col_name: str, n: int) -> list[dict[str, Any]]:
    """Top N most frequent values."""
    try:
        vc = (
            df.select(pl.col(col_name))
            .drop_nulls()
            .group_by(col_name)
            .len()
            .sort("len", descending=True)
            .head(n)
        )
        return [{"value": row[col_name], "count": row["len"]} for row in vc.iter_rows(named=True)]
    except Exception:
        return []


def _compute_histogram(df: pl.DataFrame, col_name: str, bin_count: int) -> list[dict[str, Any]]:
    try:
        hist_df = df.select(
            pl.col(col_name).hist(bin_count=bin_count, include_breakpoint=True)
        ).unnest(col_name)

        return [
            {
                "breakpoint": row["breakpoint"],
                "count": row["count"],
            }
            for row in hist_df.iter_rows(named=True)
        ]
    except Exception:
        return []


def _ordered_series(df: pl.DataFrame, col_name: str, config: ProfileConfig) -> pl.Series:
    """Return a float series, optionally sorted by ts_sortby before extraction.

    Only used for numeric TS analysis. Datetime analysis keeps its own ordering.
    Falls back silently to natural row order when the sort column is missing or
    not usable.
    """
    source = df
    sortby = config.ts_sortby
    if sortby is not None and sortby in df.columns:
        try:
            source = df.sort(sortby)
        except Exception:
            source = df
    return source.select(pl.col(col_name).drop_nulls()).get_column(col_name).cast(pl.Float64)


def _detect_timeseries(df: pl.DataFrame, col_name: str, config: ProfileConfig) -> bool:
    """Detect time dependence via lagged autocorrelation (Polars-native)."""
    if not config.ts_active:
        return False

    vals = _ordered_series(df, col_name, config)
    n = vals.len()
    if n < 3:
        return False

    # Low-cardinality columns (yr/season/holiday/workingday) are categorical in
    # ydata and cannot meaningfully be time series: their values repeat so often
    # that any lag autocorrelates, but the signal carries no information. We use
    # distinct < 6 so mnth (12) and weekday (7) still qualify as TS, matching
    # ydata's numeric TS list on the bike hour dataset.
    distinct = df.select(pl.col(col_name).n_unique()).item()
    if distinct < 6:
        return False

    for lag in config.ts_lags:
        if lag >= n:
            continue
        orig = vals.slice(0, n - lag)
        shifted = vals.slice(lag, n - lag)
        ac = _pearson_corr(orig, shifted)
        if ac is not None and ac >= config.ts_autocorrelation_threshold:
            return True

    return False


def _pearson_corr(a: pl.Series, b: pl.Series) -> float | None:
    """Pearson correlation between two equal-length slices (pandas autocorr parity).

    pandas Series.autocorr(lag) computes corr(series[lag:], series[:-lag]),
    which uses each slice's own mean and std. This mirrors that exactly and
    returns None when either slice has zero variance (pandas yields NaN).
    """
    n = a.len()
    if n < 2:
        return None

    a_mean = a.mean()
    b_mean = b.mean()
    a_std = a.std(ddof=0)
    b_std = b.std(ddof=0)
    if a_mean is None or b_mean is None or a_std is None or b_std is None:
        return None
    if a_std <= 0 or b_std <= 0:
        return None

    cov = ((a - a_mean) * (b - b_mean)).sum() / n
    return float(cov / (a_std * b_std))


def _adf_stationarity(
    df: pl.DataFrame, col_name: str, config: ProfileConfig
) -> tuple[float | None, float | None, bool]:
    """Augmented Dickey-Fuller test via statsmodels.

    Returns (statistic, p_value, is_stationary).
    """
    vals = _ordered_series(df, col_name, config)
    n = vals.len()
    if n < 5:
        return None, None, False

    if config.ts_adf_max_points is not None and n > config.ts_adf_max_points:
        step = n / config.ts_adf_max_points
        idxs = [int(i * step) for i in range(config.ts_adf_max_points)]
        vals = vals.gather(idxs)
        n = vals.len()

    try:
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(
            vals.to_numpy(),
            autolag=config.ts_adf_autolag,
            maxlag=config.ts_adf_maxlag,
        )
        statistic = float(result[0])
        p_value = float(result[1])
    except Exception:
        return None, None, False

    return statistic, p_value, p_value < config.ts_significance


def _extract_line_data(df: pl.DataFrame, col_name: str, config: ProfileConfig) -> list[float]:
    """Uniform-sample numeric series for a lightweight line plot."""
    vals = _ordered_series(df, col_name, config)
    n = vals.len()
    if n == 0:
        return []
    max_points = config.ts_line_max_points
    if n <= max_points:
        return [round(v, 4) for v in vals.to_list()]
    step = n / max_points
    idxs = [int(i * step) for i in range(max_points)]
    sampled = vals.gather(idxs)
    return [round(v, 4) for v in sampled.to_list()]


def _compute_acf_pacf(
    df: pl.DataFrame, col_name: str, config: ProfileConfig
) -> tuple[list[float], list[float]]:
    """Compute ACF and PACF via statsmodels for a time-series column."""
    vals = _ordered_series(df, col_name, config)
    n = vals.len()
    if n < 3:
        return [], []

    if config.ts_acf_pacf_max_points is not None and n > config.ts_acf_pacf_max_points:
        step = n / config.ts_acf_pacf_max_points
        idxs = [int(i * step) for i in range(config.ts_acf_pacf_max_points)]
        vals = vals.gather(idxs)
        n = vals.len()

    nlags = min(config.ts_pacf_acf_lag, n - 2)
    if nlags < 1:
        return [], []

    try:
        from statsmodels.tsa.stattools import acf, pacf

        x = vals.to_numpy()
        acf_vals = acf(x, nlags=nlags, fft=True)
        pacf_vals = pacf(x, nlags=nlags, method="ywm")
    except Exception:
        return [], []

    return (
        [round(float(v), 4) for v in acf_vals],
        [round(float(v), 4) for v in pacf_vals],
    )


def _detect_seasonality(
    df: pl.DataFrame, col_name: str, config: ProfileConfig
) -> tuple[bool, list[float]]:
    """Detect periodic seasonality via FFT power-spectrum peak detection."""
    vals = _ordered_series(df, col_name, config)
    n = vals.len()
    if n < 16:
        return False, []

    # Low-cardinality columns (e.g. season with 4 values, holiday binary) cannot
    # carry a meaningful Fourier period — the spectrum reports spurious peaks
    # that never represent real seasonality. Matches _detect_timeseries: columns
    # with < 6 distinct values never reach the FFT stage.
    distinct = df.select(pl.col(col_name).n_unique()).item()
    if distinct < 6:
        return False, []

    try:
        import numpy as np
        from scipy.signal import find_peaks, periodogram

        x = vals.to_numpy()

        # One-sided spectrum with linear detrend and a rectangular window.
        # periodogram keeps the scaling consistent and avoids computing the
        # symmetric negative frequencies of a real-valued signal.
        freq, psd = periodogram(
            x,
            fs=1.0,
            window="boxcar",
            detrend="linear",
            return_onesided=True,
            scaling="spectrum",
        )

        pos = (freq > 0) & (freq > (2.0 / n))
        freq = freq[pos]
        psd_pos = psd[pos]
        total_power = float(psd_pos.sum())
        # Detrended flat signals (e.g. a pure linear trend) leave only a
        # near-zero numerical residue; treat them as non-seasonal instead of
        # letting that residue dominate the power ratio.
        if total_power <= 1e-8:
            return False, []

        peak_indices, _ = find_peaks(psd_pos)
        if len(peak_indices) == 0:
            return False, []

        peak_indices = peak_indices[np.argsort(psd_pos[peak_indices])[::-1]]
        dominant = psd_pos[peak_indices[0]] / total_power
        if dominant < config.ts_seasonality_power_threshold:
            return False, []

        periods = _harmonic_filtered_periods(freq, peak_indices)
        return True, periods
    except Exception:
        return False, []


def _harmonic_filtered_periods(freq: Any, peak_indices: Any) -> list[float]:
    """Keep fundamental periods, dropping near-integer multiples of stronger peaks.

    Mirrors ydata's harmonic cleaning: a peak whose frequency is a near-integer
    multiple of an already-kept stronger peak is considered a harmonic.
    """
    kept: list[float] = []
    for idx in peak_indices[:3]:
        f = float(freq[idx])
        if f <= 0:
            continue
        is_harmonic = False
        for base_f in kept:
            ratio = f / base_f
            fraction = abs(ratio - round(ratio))
            if fraction < 0.01:
                is_harmonic = True
                break
        if not is_harmonic:
            kept.append(f)

    return [round(float(1.0 / f), 2) for f in kept]


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _empty_stats(col_name: str) -> NumericStats:
    return NumericStats(
        column_name=col_name,
        column_type=ColumnType.NUMERIC,
        count=0,
        missing_count=0,
        missing_pct=0.0,
    )