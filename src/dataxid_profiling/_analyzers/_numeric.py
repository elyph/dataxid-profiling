from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from dataxid_profiling._analyzers import NumericStats
from dataxid_profiling._type_inference import ColumnType

if TYPE_CHECKING:
    from dataxid_profiling._config import ProfileConfig

# Ordinal/categorical integer codes (season, yr, holiday, workingday,
# weathersit) live in a small dense range and repeat by design. They would
# trivially pass a lag autocorrelation test, but the pattern carries no
# time-series information. Detection is data-adaptive: an integer column is an
# ordinal code only when it has few distinct values AND those values fall in a
# small, dense, low-magnitude range. A measurement with few values but a wide
# spread (e.g. [0, 10, 1000, 5000]) stays eligible for TS analysis.
_ORDINAL_MAX_DISTINCT = 5
_ORDINAL_MAX_ABS_VALUE = 100
_ORDINAL_DENSE_RANGE_SLACK = 2

# Time-series detection minimums. ADF needs 5 points for a stable regression,
# ACF/PACF need 3 for a meaningful lag, and FFT seasonality needs enough
# points to resolve a low-frequency peak without a spurious boundary bin.
_MIN_POINTS_TIMESERIES = 3
_MIN_POINTS_ADF = 5
_MIN_POINTS_ACF_PACF = 3
_MIN_POINTS_SEASONALITY = 16

# Default lag list is hourly-oriented (1h, 7h, 12h, 24h, 30h); when a sampling
# interval is known we also probe one day and one week in sample units.
_BASE_TS_LAGS = (1, 7, 12, 24, 30)
_DAY_SECONDS = 86_400
_WEEK_SECONDS = 604_800

# FFT seasonality: a one-sided spectrum with a rectangular window and linear
# detrend; keep only frequencies above 2/n (one full cycle in the window) and
# treat a near-zero total power as a flat, non-seasonal signal.
_MIN_FREQ_CYCLES = 2.0
_SEASONAL_TOTAL_POWER_EPS = 1e-8

# Harmonic cleaning keeps at most the top 3 peaks and drops a peak whose
# frequency is within this fraction of a near-integer multiple of a kept peak.
_MAX_PERIOD_PEAKS = 3
_HARMONIC_RATIO_TOLERANCE = 0.01


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
    line_data, line_x = _extract_line_xy(df, col_name, config) if is_ts else ([], [])
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
        line_x=line_x,
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
    if n < _MIN_POINTS_TIMESERIES:
        return False

    if _is_ordinal_code(df, col_name):
        return False

    lags = config.ts_lags if config.ts_lags is not None else _derive_ts_lags(df, config)
    for lag in lags:
        if lag >= n:
            continue
        orig = vals.slice(0, n - lag)
        shifted = vals.slice(lag, n - lag)
        ac = _pearson_corr(orig, shifted)
        if ac is not None and ac >= config.ts_autocorrelation_threshold:
            return True

    return False


def _is_ordinal_code(df: pl.DataFrame, col_name: str) -> bool:
    """True when an integer column is an ordinal code, not a time series.

    Ordinal codes (season, yr, holiday, workingday, weathersit) are integers in
    a small, dense, low-magnitude range. This adapts to the observed values
    rather than a fixed distinct-count cutoff: a 4-value column spanning
    0..3 is a code, but a 4-value column like [0, 10, 1000, 5000] is a
    measurement and remains eligible for time-series analysis.
    """
    series = df.select(pl.col(col_name).drop_nulls()).get_column(col_name)
    if not series.dtype.is_integer():
        return False

    distinct = series.n_unique()
    if distinct > _ORDINAL_MAX_DISTINCT:
        return False

    min_val = series.min()
    max_val = series.max()
    if min_val is None or max_val is None:
        return False
    if abs(min_val) > _ORDINAL_MAX_ABS_VALUE or abs(max_val) > _ORDINAL_MAX_ABS_VALUE:
        return False

    # Dense range: the count of integers covered by [min, max] is close to the
    # number of distinct values. season (1..4) has span 4 == distinct 4.
    span = int(max_val) - int(min_val) + 1
    return span <= distinct + _ORDINAL_DENSE_RANGE_SLACK


def _derive_ts_lags(df: pl.DataFrame, config: ProfileConfig) -> tuple[int, ...]:
    """Derive a lag list from the sampling interval when no explicit list is set.

    Defaults to the original hourly-oriented lags; if ts_sortby points at a
    datetime column, add daily and weekly lags derived from its median sampling
    interval (e.g. 15-minute data gets 96 and 672).
    """
    base = _BASE_TS_LAGS
    sortby = config.ts_sortby
    if sortby is None or sortby not in df.columns:
        return base

    try:
        col = pl.col(sortby)
        intervals = (
            df.select(col)
            .drop_nulls()
            .select(col.diff().dt.total_seconds().abs().drop_nulls())
        )
        median = intervals.select(pl.col(sortby).median()).item()
    except Exception:
        return base

    if not median or median <= 0:
        return base

    extra: list[int] = []
    for target_seconds in (_DAY_SECONDS, _WEEK_SECONDS):
        lag = round(target_seconds / median)
        if lag not in base and lag >= 1:
            extra.append(lag)
    return tuple(base) + tuple(extra)


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
    if n < _MIN_POINTS_ADF:
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


def _extract_line_xy(
    df: pl.DataFrame, col_name: str, config: ProfileConfig
) -> tuple[list[float], list[str]]:
    """Uniform-sample (x, y) pairs for a time-series line plot.

    The y values are the numeric series; the x values are real timestamps when
    ``ts_sortby`` points at a datetime column, otherwise positional indices.
    Both axes come from the same rows and the same ``ts_line_max_points``
    sampling so the plot stays aligned.
    """
    source = df
    sortby = config.ts_sortby
    if sortby is not None and sortby in df.columns:
        try:
            source = df.sort(sortby)
        except Exception:
            source = df

    source = source.filter(pl.col(col_name).is_not_null())
    n = source.height
    if n == 0:
        return [], []

    y_vals = source.get_column(col_name).cast(pl.Float64)
    has_time_x = sortby is not None and sortby in source.columns

    max_points = config.ts_line_max_points
    if n > max_points:
        step = n / max_points
        idxs = [int(i * step) for i in range(max_points)]
        y_sampled = y_vals.gather(idxs)
        x_series = (
            source.get_column(sortby).gather(idxs)
            if has_time_x
            else pl.Series([i for i in idxs])
        )
    else:
        y_sampled = y_vals
        x_series = source.get_column(sortby) if has_time_x else pl.Series([i for i in range(n)])

    y = [round(v, 4) for v in y_sampled.to_list()]
    x = _format_timestamps(x_series) if has_time_x else [str(v) for v in x_series.to_list()]
    return y, x


def _format_timestamps(series: pl.Series) -> list[str]:
    """Render datetime values as compact string labels for a category axis."""
    try:
        if isinstance(series.dtype, pl.Datetime):
            return [
                v.strftime("%Y-%m-%d %H:%M") if v is not None else ""
                for v in series.to_list()
            ]
        return [str(v) for v in series.to_list()]
    except Exception:
        return [str(v) for v in series.to_list()]


def _compute_acf_pacf(
    df: pl.DataFrame, col_name: str, config: ProfileConfig
) -> tuple[list[float], list[float]]:
    """Compute ACF and PACF via statsmodels for a time-series column."""
    vals = _ordered_series(df, col_name, config)
    n = vals.len()
    if n < _MIN_POINTS_ACF_PACF:
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
    if vals.len() < _MIN_POINTS_SEASONALITY:
        return False, []

    try:
        freq, psd = _periodogram_spectrum(vals)
        peak_indices, snr = _seasonal_peaks(freq, psd)
    except Exception:
        return False, []

    if peak_indices is None or snr < config.ts_seasonality_snr_threshold:
        return False, []

    return True, _harmonic_filtered_periods(freq, peak_indices)


def _periodogram_spectrum(vals: pl.Series) -> tuple[Any, Any]:
    """Compute the positive one-sided power spectrum of a numeric series.

    A rectangular window plus linear detrend keeps the scaling consistent and
    avoids computing the symmetric negative frequencies of a real signal.
    Returns the positive frequencies and their PSD, or (None, None) when the
    series is flat.
    """
    from scipy.signal import periodogram

    n = vals.len()
    freq, psd = periodogram(
        vals.to_numpy(),
        fs=1.0,
        window="boxcar",
        detrend="linear",
        return_onesided=True,
        scaling="spectrum",
    )

    pos = (freq > 0) & (freq > (_MIN_FREQ_CYCLES / n))
    freq = freq[pos]
    psd_pos = psd[pos]
    # Detrended flat signals (e.g. a pure linear trend) leave only a near-zero
    # numerical residue; return None so the caller treats them as non-seasonal.
    if float(psd_pos.sum()) <= _SEASONAL_TOTAL_POWER_EPS:
        return None, None
    return freq, psd_pos


def _seasonal_peaks(freq: Any, psd: Any) -> tuple[Any, float]:
    """Return candidate peak indices (strongest first) and their best SNR.

    Peaks are found on the positive spectrum; the signal-to-noise ratio is the
    strongest peak against the spectral median. White noise has max/median ~
    ln(n_bins), so a real periodic signal stands far above it. Using the median
    (not total power) stops broadband power from drowning out a weak but
    coherent daily/weekly peak.
    """
    import numpy as np
    from scipy.signal import find_peaks

    if freq is None or psd is None:
        return None, 0.0

    noise_floor = float(np.median(psd))
    if noise_floor <= 0:
        return None, 0.0

    peak_indices, _ = find_peaks(psd)
    if len(peak_indices) == 0:
        return None, 0.0

    peak_indices = peak_indices[np.argsort(psd[peak_indices])[::-1]]
    snr = float(psd[peak_indices[0]] / noise_floor)
    return peak_indices, snr


def _harmonic_filtered_periods(freq: Any, peak_indices: Any) -> list[float]:
    """Keep fundamental periods, dropping near-integer multiples of stronger peaks.

    Mirrors ydata's harmonic cleaning: a peak whose frequency is a near-integer
    multiple of an already-kept stronger peak is considered a harmonic.
    """
    kept: list[float] = []
    for idx in peak_indices[:_MAX_PERIOD_PEAKS]:
        f = float(freq[idx])
        if f <= 0:
            continue
        is_harmonic = False
        for base_f in kept:
            ratio = f / base_f
            fraction = abs(ratio - round(ratio))
            if fraction < _HARMONIC_RATIO_TOLERANCE:
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