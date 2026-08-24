"""HTML report rendering via Jinja2."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from dataxid_profiling._alerts import Alert  # noqa: TC001 — used at runtime
from dataxid_profiling._analyzers import (
    BooleanStats,
    CategoricalStats,
    ColumnStats,
    DatetimeStats,
    NumericStats,
)
from dataxid_profiling._correlations import CorrelationResult  # noqa: TC001
from dataxid_profiling._dataset_overview import DatasetOverview  # noqa: TC001 — used at runtime
from dataxid_profiling._interactions import InteractionData  # noqa: TC001
from dataxid_profiling._report._charts import ChartRenderer, EChartsRenderer

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def render_html(
    *,
    title: str,
    version: str,
    overview: DatasetOverview,
    column_stats: dict[str, ColumnStats],
    alerts: list[Alert],
    correlations: dict[str, CorrelationResult],
    interactions: InteractionData | None = None,
    chart_renderer: ChartRenderer | None = None,
    time_index: dict[str, Any] | None = None,
) -> str:
    renderer = chart_renderer or EChartsRenderer()
    env = _build_env()
    template = env.get_template("report.html.j2")

    columns = _prepare_columns(column_stats, renderer)
    correlation_charts = _prepare_correlation_charts(correlations, renderer)
    missing_bar_chart = _prepare_missing_bar_chart(overview, renderer)
    time_index_html = _prepare_time_index_charts(time_index, renderer)
    interactions_payload = _prepare_interactions(interactions)
    alert_dicts = [
        {
            "column": a.column,
            "alert_type": a.alert_type.name,
            "value": a.value,
            "details": a.details,
        }
        for a in alerts
    ]

    ov_dict = asdict(overview)

    return template.render(
        title=title,
        version=version,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        overview=ov_dict,
        columns=columns,
        alerts=alert_dicts,
        correlation_charts=correlation_charts,
        interactions=interactions_payload,
        missing_bar_chart=missing_bar_chart,
        time_index=time_index,
        time_index_html=time_index_html,
        logo_b64=_load_asset_b64("dataxid_logo.png"),
        icon_b64=_load_asset_b64("icon.png"),
    )


def _load_asset_b64(filename: str) -> str:
    path = _TEMPLATE_DIR / filename
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _build_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
    )
    env.filters["format_number"] = _format_number
    env.filters["format_pct"] = _format_pct
    env.filters["format_float"] = _format_float
    env.filters["format_alert_value"] = _format_alert_value
    return env


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _format_pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        pct = float(value) * 100
        if 0 < pct < 0.1:
            return "<0.1%"
        return f"{pct:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_float(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_alert_value(alert_type: str, value: Any) -> str:
    """Format an alert value based on its type.

    Percentage-based alerts keep the percent format; time-series and count
    alerts are rendered as plain numbers so a second value is not shown as a
    misleading percentage.
    """
    if value is None:
        return "—"

    raw_alerts = {
        "UNSORTED_DATES",
        "IRREGULAR_INTERVALS",
        "LARGE_GAPS",
        "NON_STATIONARY",
        "SEASONAL",
        "CONSTANT",
    }
    if alert_type in raw_alerts:
        try:
            return f"{float(value):,.3f}"
        except (TypeError, ValueError):
            return str(value)

    return _format_pct(value)


def _prepare_columns(
    column_stats: dict[str, ColumnStats],
    renderer: ChartRenderer,
) -> dict[str, dict[str, Any]]:
    columns: dict[str, dict[str, Any]] = {}
    for idx, (col_name, stats) in enumerate(column_stats.items()):
        col_dict = asdict(stats)
        col_dict["column_type"] = stats.column_type.name
        col_dict["chart_html"] = _chart_for_column(stats, renderer, idx)
        col_dict["wordcloud_html"] = _wordcloud_for_column(stats, renderer, idx)
        col_dict["ts_chart_html"] = _ts_chart_for_column(stats, renderer, idx)
        col_dict["acf_chart_html"], col_dict["pacf_chart_html"] = _acf_pacf_charts_for_column(
            stats, renderer, idx
        )
        col_dict["gap_chart_html"] = _gap_chart_for_column(stats, renderer, idx)
        columns[col_name] = col_dict
    return columns


def _chart_for_column(stats: ColumnStats, renderer: ChartRenderer, idx: int) -> str:
    div_id = f"col_chart_{idx}"

    if isinstance(stats, NumericStats) and stats.histogram and not stats.is_timeseries:
        labels = [str(round(h["breakpoint"], 2)) for h in stats.histogram]
        values = [h["count"] for h in stats.histogram]
        return renderer.histogram(div_id, labels, values, title="Distribution")

    if isinstance(stats, CategoricalStats) and stats.top_values:
        labels = [str(tv["value"]) for tv in stats.top_values]
        values = [tv["count"] for tv in stats.top_values]
        if stats.other_values is not None:
            labels.append("Other")
            values.append(stats.other_values.count)
        return renderer.bar_horizontal(div_id, labels, values, title="Top Values")

    if isinstance(stats, BooleanStats):
        labels, values = [], []
        if stats.true_count > 0:
            labels.append("True")
            values.append(stats.true_count)
        if stats.false_count > 0:
            labels.append("False")
            values.append(stats.false_count)
        if stats.missing_count > 0:
            labels.append("Missing")
            values.append(stats.missing_count)
        if labels:
            return renderer.pie(div_id, labels, values, title="Distribution")

    return ""


def _ts_chart_for_column(stats: ColumnStats, renderer: ChartRenderer, idx: int) -> str:
    if not isinstance(stats, NumericStats) or not stats.is_timeseries or not stats.line_data:
        return ""
    x = stats.line_x or [str(i) for i in range(len(stats.line_data))]
    return renderer.line(f"col_ts_{idx}", x, stats.line_data, title="Time Series")


def _acf_pacf_charts_for_column(
    stats: ColumnStats, renderer: ChartRenderer, idx: int
) -> tuple[str, str]:
    if not isinstance(stats, NumericStats) or not stats.acf_values:
        return "", ""
    labels = [str(i) for i in range(len(stats.acf_values))]
    acf_chart = renderer.histogram(f"col_acf_{idx}", labels, stats.acf_values, title="ACF")
    pacf_chart = renderer.histogram(f"col_pacf_{idx}", labels, stats.pacf_values, title="PACF")
    return acf_chart, pacf_chart


def _gap_chart_for_column(stats: ColumnStats, renderer: ChartRenderer, idx: int) -> str:
    if not isinstance(stats, DatetimeStats) or not stats.interval_values:
        return ""
    labels = [str(i) for i in range(len(stats.interval_values))]
    return renderer.gap_plot(
        f"col_gap_{idx}",
        labels,
        stats.interval_values,
        stats.gap_indices,
        threshold=stats.gap_threshold_seconds,
        title="Sampling Intervals & Gaps",
    )


def _wordcloud_for_column(stats: ColumnStats, renderer: ChartRenderer, idx: int) -> str:
    if not isinstance(stats, CategoricalStats) or not stats.top_values:
        return ""
    words = [str(tv["value"]) for tv in stats.top_values]
    weights = [tv["count"] for tv in stats.top_values]
    return renderer.word_cloud(f"col_wc_{idx}", words, weights, title="Word Cloud")


def _prepare_time_index_charts(
    time_index: dict[str, Any] | None,
    renderer: ChartRenderer,
) -> dict[str, str]:
    if not time_index:
        return {"original": "", "scaled": ""}

    names = time_index.get("series_names", [])
    x = time_index.get("x", [])
    original = time_index.get("original", {})
    scaled = time_index.get("scaled", {})

    return {
        "original": renderer.multi_line(
            "ts_overview_original",
            names,
            x,
            original,
            title="Time Series Overview — Original",
        ),
        "scaled": renderer.multi_line(
            "ts_overview_scaled",
            names,
            x,
            scaled,
            title="Time Series Overview — Scaled",
        ),
    }


def _prepare_missing_bar_chart(
    overview: DatasetOverview,
    renderer: ChartRenderer,
) -> str:
    missing = overview.missing_per_column
    if not missing:
        return ""
    cols_with_missing = {col: info for col, info in missing.items() if info["count"] > 0}
    if not cols_with_missing:
        return ""
    labels = list(cols_with_missing.keys())
    values = [info["count"] for info in cols_with_missing.values()]
    return renderer.bar_horizontal(
        "missing_bar", labels, values, title="Missing Values per Column"
    )


_SYMMETRIC_RANGE: dict[str, tuple[float, float]] = {
    "pearson": (-1.0, 1.0),
    "spearman": (-1.0, 1.0),
    "kendall": (-1.0, 1.0),
    "cramers_v": (0.0, 1.0),
    "phik": (0.0, 1.0),
}


def _prepare_correlation_charts(
    correlations: dict[str, CorrelationResult],
    renderer: ChartRenderer,
) -> list[dict[str, str]]:
    """Build one heatmap per correlation method. Returns [{name, div_id, chart_html}, ...]."""
    charts: list[dict[str, str]] = []
    for method, cr in correlations.items():
        matrix = cr.matrix
        labels = matrix["column"].to_list()
        data: list[list[float]] = []
        for row in matrix.iter_rows(named=True):
            data.append([float(row[col]) for col in labels])

        div_id = f"corr_{method}"
        display_name = method.replace("_", " ").title()
        title_str = f"{display_name} Correlation"
        vrange = _SYMMETRIC_RANGE.get(method)
        chart_html = renderer.heatmap(
            div_id,
            labels,
            labels,
            data,
            title=title_str,
            value_range=vrange,
        )
        charts.append({"name": display_name, "div_id": div_id, "chart_html": chart_html})
    return charts


def _prepare_interactions(
    interactions: InteractionData | None,
) -> dict[str, Any] | None:
    """Serialize InteractionData to a template-friendly dict with JSON payloads."""
    if interactions is None:
        return None

    boxplot_serialized: dict[str, dict[str, list[dict]]] = {}
    for cat_col, num_map in interactions.boxplot_stats.items():
        boxplot_serialized[cat_col] = {}
        for num_col, groups in num_map.items():
            boxplot_serialized[cat_col][num_col] = [asdict(g) for g in groups]

    return {
        "numeric_columns": interactions.numeric_columns,
        "categorical_columns": interactions.categorical_columns,
        "numeric_data_json": json.dumps(
            interactions.numeric_data,
            ensure_ascii=False,
        ),
        "boxplot_stats_json": json.dumps(
            boxplot_serialized,
            ensure_ascii=False,
        ),
        "sampled": interactions.sampled,
        "total_rows": interactions.total_rows,
        "sample_size": interactions.sample_size,
    }
