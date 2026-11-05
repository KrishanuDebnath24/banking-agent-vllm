"""Charts under answers.

A chart comes only from the chart server's tools (mcp_servers/charts/server.py), which draw figures
the transactions server worked out in code and saved. The web API copies the chart from the tool's
result as the answer runs, never from the model's words, so a chart can't show a figure the bank
didn't give. This file checks and trims what it copies; the page draws it (web/app.js).
"""

from __future__ import annotations

import json
import re

CHART_TOOLS = {"column_chart", "line_chart", "stacked_column_chart", "bar_chart", "donut_chart", "combo_chart"}
FORMS = {"column", "line", "stacked", "bar", "donut", "combo"}
MAX_CHARTS = 3            # per answer
MAX_LABELS = {"month": 24, "name": 13, "day": 62, "dayofmonth": 31}
TIME_AXES = ("month", "day", "dayofmonth")   # columns or lines; "name" is bars or a donut
MAX_SERIES = 4
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")


def _short(value, limit: int) -> str | None:
    text = " ".join(str(value or "").split())   # no line breaks or control characters
    return text[:limit] or None


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value < 0:
        return None
    return round(float(value), 2)


def _count(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def check(chart) -> dict | None:
    """The chart, checked and trimmed; None if anything is off (draw nothing rather than something wrong)."""
    if not isinstance(chart, dict) or chart.get("form") not in FORMS:
        return None
    axis = chart.get("axis") if chart.get("axis") in TIME_AXES else "name"
    if (axis in TIME_AXES) != (chart["form"] in ("column", "line", "stacked", "combo")):
        return None
    if (chart["form"] == "stacked" and axis != "month") or (axis == "dayofmonth" and chart["form"] not in ("line", "combo")):
        return None
    combo = chart["form"] == "combo"   # columns for each day, then 1-2 running-total lines ("line", "dashed")
    if combo and axis == "month":
        return None
    labels = chart.get("labels")
    if not isinstance(labels, list) or not 2 <= len(labels) <= MAX_LABELS[axis]:
        return None
    if axis == "month" and not all(isinstance(m, str) and _MONTH.match(m) for m in labels):
        return None
    if axis == "day" and not all(isinstance(d, str) and _DATE.match(d) for d in labels):
        return None
    if axis == "dayofmonth" and labels != [str(d) for d in range(1, len(labels) + 1)]:
        return None
    running = axis == "dayofmonth" and chart.get("running") is True   # running totals: they only go up
    labels = [_short(x, 60) for x in labels] if axis == "name" else labels
    if None in labels:
        return None
    raw_series = chart.get("series")
    if not isinstance(raw_series, list) or not 1 <= len(raw_series) <= MAX_SERIES:
        return None
    if chart["form"] in ("bar", "donut") and len(raw_series) != 1:
        return None
    if chart["form"] == "stacked" and len(raw_series) < 2:
        return None
    if chart["form"] == "line" and len(raw_series) > 3:   # more lines than colours that can be told apart
        return None
    if combo and (not 2 <= len(raw_series) <= 3
                  or [s.get("kind") if isinstance(s, dict) else None for s in raw_series][:1] != ["bar"]
                  or not all(isinstance(s, dict) and s.get("kind") in ("line", "dashed") for s in raw_series[1:])):
        return None
    series = []
    for s in raw_series:
        values = s.get("values") if isinstance(s, dict) else None
        if not isinstance(values, list) or len(values) != len(labels):
            return None
        values = [_number(v) for v in values]
        if None in values:
            return None
        counts = s.get("counts")
        counts = [_count(c) for c in counts] if isinstance(counts, list) and len(counts) == len(labels) else None
        kind = s.get("kind") if combo else None
        if (running or kind in ("line", "dashed")) and any(b < a for a, b in zip(values, values[1:])):
            return None
        entry = {"name": _short(s.get("name"), 40) or "Figures", "values": values, "counts": counts}
        series.append({**entry, "kind": kind} if combo else entry)
    notes = chart.get("notes") if isinstance(chart.get("notes"), dict) else {}
    notes = {k: _short(v, 60) for k, v in notes.items() if k in labels and _short(v, 60)}
    period = chart.get("period") if isinstance(chart.get("period"), dict) else {}
    currency = str(chart.get("currency") or "GBP")
    return {"form": chart["form"], "title": _short(chart.get("title"), 60) or "Chart",
            "money": "in" if chart.get("money") == "in" else "out",
            "currency": currency if _CURRENCY.match(currency) else "GBP",
            "axis": axis, "by": "month" if axis == "month" else "day" if axis in TIME_AXES else (
                "payee" if chart.get("by") == "payee" else "category"),
            "labels": labels, "series": series, "notes": notes, "running": running,
            "note": _short(chart.get("note"), 160),
            # running totals: the total so far is the first line's last point, not a sum of points
            "total": series[0]["values"][-1] if running else round(sum(series[0]["values"]), 2) if combo
            else round(sum(v for s in series for v in s["values"]), 2),
            "period": {k: period[k] for k in ("from", "to") if isinstance(period.get(k), str) and _DATE.match(period[k])}}


def from_tool(name: str, content: str) -> dict | None:
    """The chart in a chart tool's result, or None (another tool, an error, or anything odd)."""
    if name not in CHART_TOOLS:
        return None
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("error"):
        return None
    return check(data.get("chart"))
