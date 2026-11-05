"""Charts MCP server: draws figures the transactions server already worked out and saved.

It has no bank access at all. Each tool takes a dataset id (ds-...), opens that dataset if it
belongs to the signed-in customer (mcp_servers/datasets.py), and returns a chart for the page to
draw, plus a one-line summary for the chart specialist to report. The figures in a chart therefore
come only from the bank server's code, never from the AI, and the same dataset can be drawn again
in another form without asking the bank.

Which form fits which figures:
    month-by-month totals (1-4 sets of figures)    column_chart, line_chart (up to 3), stacked_column_chart (2-4)
    day-by-day totals (up to 62 days)               column_chart, line_chart, combo_chart (+ running total)
    month to date against the same days before      line_chart (both months' running totals), combo_chart
                                                    (a column per day, this month's running total as a
                                                    line, last month's as a dashed line)
    a breakdown by category or payee                bar_chart (ranked), donut_chart (shares, up to 6)
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import Context, FastMCP

from mcp_servers import datasets
from mcp_servers.common import PORTS, customer, error, transport_security

mcp = FastMCP("charts", port=PORTS["charts"], stateless_http=True, json_response=True,
              transport_security=transport_security(),
              instructions="Draws charts of figures the bank has saved (dataset ids like ds-4f2a9c1e).")

MONTH_FORMS = ("column_chart", "line_chart", "stacked_column_chart")
BREAKDOWN_FORMS = ("bar_chart", "donut_chart")
DAY_FORMS = ("column_chart", "line_chart", "combo_chart")
FITS = {"monthly": MONTH_FORMS, "breakdown": BREAKDOWN_FORMS, "daily": DAY_FORMS,
        "pace": ("line_chart", "combo_chart")}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _money(value: float, currency: str) -> str:
    return f"£{value:,.2f}" if currency == "GBP" else f"{value:,.2f} {currency}"


def _month(key: str) -> str:
    year, month = key.split("-")
    return f"{MONTHS[int(month) - 1]} {year}"


async def _open(ctx, dataset_id: str) -> tuple[dict | None, str | None]:
    """(dataset, None) or (None, an error to return)."""
    try:
        cust = await customer(ctx)
    except PermissionError as e:
        return None, error("not_authorised", detail=str(e))
    found = datasets.load(cust.records_key, dataset_id)
    if found is None:
        return None, error("not_found", message="No saved figures with that id for this customer (they may "
                                                 "have expired). Ask for the figures again.")
    return found, None


def _wrong(form: str, data: dict) -> str:
    kind = data["kind"]
    what = {"monthly": "month-by-month totals", "daily": "day-by-day totals",
            "pace": "running totals of this month and the same days of last month"}.get(
        kind, "a breakdown by " + data.get("by", "category"))
    return error("wrong_form", message=f"{form} doesn't fit {what}.", forms_that_fit=list(FITS.get(kind, ())))


def _day(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{int(d)} {MONTHS[int(m) - 1]} {y}"


def _daily(form: str, data: dict, title: str | None) -> str:
    s, days = data["series"][0], data["days"]
    cur, period = data["currency"], f"{data['period']['from']} to {data['period']['to']}"
    if len(days) < 2:
        return error("too_little", message=f"Only one day, so there's no chart: give the figure in words "
                                           f"instead. {s['name']} {_money(s['total'], cur)}, {period}.")
    chart = {"form": form, "title": _title(title, data["label"]), "money": data["money"], "currency": cur,
             "axis": "day", "by": "day", "labels": days,
             "series": [{"name": s["name"], "values": s["by_day"], "counts": s["payments"]}],
             "notes": {}, "note": data.get("note"), "total": s["total"], "period": data["period"],
             "dataset_id": data["dataset_id"]}
    top = max(range(len(days)), key=lambda i: s["by_day"][i])
    kind = "column chart" if form == "column" else "line chart"
    summary = (f"Drew a {kind}: {chart['title']}, {_day(days[0])} to {_day(days[-1])}, total "
               f"{_money(s['total'], cur)}." + (f" Highest day: {_day(days[top])} ({_money(s['by_day'][top], cur)})."
                                               if s["total"] else " Nothing spent on these days."))
    if data.get("note"):
        summary += " " + data["note"]
    return json.dumps({"chart": chart, "summary": summary})


def _pace(data: dict, title: str | None) -> str:
    now, then = data["series"]
    cur, labels = data["currency"], data["days"]
    if len(labels) < 2:
        return error("too_little", message=f"Only day 1 so far, so there's no chart: give the figures in words "
                                           f"instead. {now['name']} {_money(now['total'], cur)}, "
                                           f"{then['name']} {_money(then['total'], cur)}.")
    chart = {"form": "line", "title": _title(title, data["label"]), "money": data["money"], "currency": cur,
             "axis": "dayofmonth", "by": "day", "running": True, "labels": labels,
             "series": [{"name": s["name"], "values": s["running"], "counts": None} for s in (now, then)],
             "notes": {}, "note": data.get("note"), "total": now["total"], "period": data["period"],
             "dataset_id": data["dataset_id"]}
    summary = (f"Drew a line chart: {chart['title']}, the running total by day of the month up to day "
               f"{labels[-1]}: {now['name']} {_money(now['total'], cur)} against {then['name']} "
               f"{_money(then['total'], cur)}.")
    return json.dumps({"chart": chart, "summary": summary})


def _combo(data: dict, title: str | None) -> str:
    """Columns for each day's spending, with the running total as a line on its own scale (right),
    and, for month-to-date figures, last month's running total over the same days as a dashed line."""
    cur = data["currency"]
    if data["kind"] == "pace":
        now, then = data["series"]
        if "daily" not in now:
            return error("not_found", message="These figures were saved before combined charts existed. "
                                               "Ask for the month-to-date figures again.")
        labels, axis = data["days"], "dayofmonth"
        bars, bar_counts, running, total = now["daily"], now["payments"], now["running"], now["total"]
        when = now["name"]
        series = [{"name": f"{now['name']}, each day", "kind": "bar", "values": bars, "counts": bar_counts},
                  {"name": f"{now['name']} so far", "kind": "line", "values": running, "counts": None},
                  {"name": f"{then['name']} so far", "kind": "dashed", "values": then["running"], "counts": None}]
    else:
        s = data["series"][0]
        labels, axis = data["days"], "day"
        bars, total = s["by_day"], s["total"]
        running, so_far = [], 0.0
        for v in bars:
            so_far += v
            running.append(round(so_far, 2))
        when = f"{_day(labels[0])} to {_day(labels[-1])}" if labels else ""
        series = [{"name": f"{s['name']}, each day", "kind": "bar", "values": bars, "counts": s["payments"]},
                  {"name": "Running total", "kind": "line", "values": running, "counts": None}]
    if len(labels) < 2:
        return error("too_little", message=f"Only one day, so there's no chart: give the figure in words "
                                           f"instead. {_money(total, cur)}.")
    chart = {"form": "combo", "title": _title(title, data["label"]), "money": data["money"], "currency": cur,
             "axis": axis, "by": "day", "labels": labels, "series": series, "notes": {}, "note": data.get("note"),
             "total": total, "period": data["period"], "dataset_id": data["dataset_id"]}
    top = max(range(len(labels)), key=lambda i: bars[i])
    day_name = (lambda k: f"day {k}") if axis == "dayofmonth" else _day
    summary = f"Drew a combined chart: {chart['title']}. "
    if data["kind"] == "pace":  # the comparison first: it's what the customer asked
        summary += (f"Days 1 to {labels[-1]}: {now['name']} {_money(total, cur)}, {then['name']} "
                    f"{_money(then['total'], cur)}. ")
    summary += (f"Columns for each day of {when} (total {_money(total, cur)}"
                + (f", highest on {day_name(labels[top])} at {_money(bars[top], cur)}" if total else "")
                + "), with the running total as a line")
    if data["kind"] == "pace":
        summary += (f", and {then['name']}'s running total over the same days as a dashed line "
                    f"({_money(then['total'], cur)} by day {labels[-1]})")
    return json.dumps({"chart": chart, "summary": summary + "."})


def _title(title: str | None, fallback: str) -> str:
    return " ".join((title or "").split())[:60] or fallback


def _monthly(form: str, data: dict, title: str | None) -> str:
    series = data["series"]
    if form == "line" and len(series) > 3:
        return error("wrong_form", message="More than 3 lines can't be told apart.",
                     forms_that_fit=["column_chart", "stacked_column_chart"])
    if form == "stacked" and len(series) < 2:
        return error("wrong_form", message="A stacked chart needs at least two categories; this has one.",
                     forms_that_fit=["column_chart", "line_chart"])
    if len(data["months"]) < 2:
        # No chart, but the figure itself, so the answer can still give it (in words, from the bank).
        cur, when = data["currency"], (_month(data["months"][0]) if data["months"] else "that period")
        figures = "; ".join(f"{s['name']} {_money(s['by_month'][0] if s['by_month'] else 0, cur)}" for s in series)
        return error("too_little", message=f"Only one month of figures ({when}), so there's no chart: give "
                                           f"the figure in words instead. {figures or 'Nothing spent'}, "
                                           f"{data['period']['from']} to {data['period']['to']}.")
    chart = {"form": form, "title": _title(title, f"{data['label']} by month"), "money": data["money"],
             "currency": data["currency"], "axis": "month", "by": "month", "labels": data["months"],
             "series": [{"name": s["name"], "values": s["by_month"], "counts": s["payments"]} for s in series],
             "notes": data.get("partial") or {}, "note": data.get("note"),
             "total": round(sum(s["total"] for s in series), 2), "period": data["period"],
             "dataset_id": data["dataset_id"]}
    cur = data["currency"]
    names = {"column": "column chart", "line": "line chart", "stacked": "stacked column chart"}
    parts = [f"Drew a {names[form]}: {chart['title']}, {_month(data['months'][0])} to {_month(data['months'][-1])}."]
    for s in series:
        top = max(range(len(s["by_month"])), key=lambda i: s["by_month"][i])
        # Each month's figure when there are few, so the answer can give them all.
        each = ("; ".join(f"{_month(m)} {_money(v, cur)}" for m, v in zip(data["months"], s["by_month"])) + "; "
                if len(data["months"]) <= 6 else "")
        parts.append(f"{s['name']}: {each}total {_money(s['total'], cur)}, highest {_month(data['months'][top])} "
                     f"({_money(s['by_month'][top], cur)}).")
    if len(series) > 1:
        parts.append(f"All together: {_money(chart['total'], cur)}.")
    if chart["notes"]:
        parts.append("Part months: " + "; ".join(f"{_month(k)} {v}" for k, v in chart["notes"].items()) + ".")
    if chart["note"]:
        parts.append(chart["note"])
    return json.dumps({"chart": chart, "summary": " ".join(parts)})


def _fold(items: list[dict], keep: int) -> list[dict]:
    """The biggest `keep` items, and everything else together as "Other"."""
    other = [i for i in items if i["name"].lower() == "other"]
    rest = [i for i in items if i["name"].lower() != "other"]
    if len(rest) + len(other) <= keep + 1:
        return rest + other
    shown, folded = rest[:keep], rest[keep:] + other
    return shown + [{"name": "Other", "total": round(sum(i["total"] for i in folded), 2),
                     "payments": sum(i["payments"] for i in folded)}]


def _breakdown(form: str, data: dict, title: str | None, keep: int) -> str:
    items = [i for i in data["items"] if i["total"] > 0]
    if len(items) < 2:
        cur, period = data["currency"], f"{data['period']['from']} to {data['period']['to']}"
        what = (f"{items[0]['name']} {_money(items[0]['total'], cur)} "
                f"({items[0]['payments']} payment{'' if items[0]['payments'] == 1 else 's'})") if items else "Nothing spent"
        return error("too_little", message=f"Only one item, so there's no chart: give the figure in words "
                                           f"instead. {what}, {period}.")
    items = _fold(items, keep)
    chart = {"form": form, "title": _title(title, data["label"]), "money": data["money"],
             "currency": data["currency"], "axis": "name", "by": data.get("by", "category"),
             "labels": [i["name"] for i in items],
             "series": [{"name": data["label"], "values": [i["total"] for i in items],
                         "counts": [i["payments"] for i in items]}],
             "notes": {}, "note": None, "total": data["total"], "period": data["period"],
             "dataset_id": data["dataset_id"]}
    cur = data["currency"]
    shares = ", ".join(f"{i['name']} {_money(i['total'], cur)} ({i['total'] / data['total'] * 100:.0f}%)"
                       for i in items[:4])
    kind = "ranked bar chart" if form == "bar" else "donut chart"
    summary = (f"Drew a {kind}: {chart['title']}, {data['period']['from']} to {data['period']['to']}, total "
               f"{_money(data['total'], cur)}. Biggest: {shares}.")
    return json.dumps({"chart": chart, "summary": summary})


@mcp.tool()
async def column_chart(ctx: Context, dataset_id: str, title: str | None = None) -> str:
    """Columns, one per month or one per day: the default for month-by-month and day-by-day
    totals. With several categories, their columns stand side by side in each month.

    Args:
        dataset_id: the id the transactions specialist reported, e.g. "ds-4f2a9c1e".
        title: optional chart title; leave empty to use the figures' own.
    """
    data, problem = await _open(ctx, dataset_id)
    if problem:
        return problem
    if data["kind"] == "daily":
        return _daily("column", data, title)
    return _monthly("column", data, title) if data["kind"] == "monthly" else _wrong("column_chart", data)


@mcp.tool()
async def line_chart(ctx: Context, dataset_id: str, title: str | None = None) -> str:
    """A line through the months or days: for trends over time; up to 3 categories as separate
    lines. The only form for a month-to-date comparison: this month's and last month's running
    totals, day by day.

    Args:
        dataset_id: the id the transactions specialist reported, e.g. "ds-4f2a9c1e".
        title: optional chart title.
    """
    data, problem = await _open(ctx, dataset_id)
    if problem:
        return problem
    if data["kind"] == "daily":
        return _daily("line", data, title)
    if data["kind"] == "pace":
        return _pace(data, title)
    return _monthly("line", data, title) if data["kind"] == "monthly" else _wrong("line_chart", data)


@mcp.tool()
async def combo_chart(ctx: Context, dataset_id: str, title: str | None = None) -> str:
    """One combined chart: a column for each day's spending, with the running total as a line on
    top. For month-to-date figures it also shows last month's running total over the same days as a
    dashed line, so the customer sees whether they're ahead or behind. Use when the customer wants
    each day together with the running total or last month's comparison in one chart.

    Args:
        dataset_id: the id of day-by-day figures or of a month-to-date comparison, e.g. "ds-4f2a9c1e".
        title: optional chart title.
    """
    data, problem = await _open(ctx, dataset_id)
    if problem:
        return problem
    return _combo(data, title) if data["kind"] in ("daily", "pace") else _wrong("combo_chart", data)


@mcp.tool()
async def stacked_column_chart(ctx: Context, dataset_id: str, title: str | None = None) -> str:
    """Columns per month with 2-4 categories stacked on top of each other: each category's share
    and the month's combined total at once.

    Args:
        dataset_id: the id of month-by-month figures for 2-4 categories.
        title: optional chart title.
    """
    data, problem = await _open(ctx, dataset_id)
    if problem:
        return problem
    return _monthly("stacked", data, title) if data["kind"] == "monthly" else _wrong("stacked_column_chart", data)


@mcp.tool()
async def bar_chart(ctx: Context, dataset_id: str, title: str | None = None, top: int = 8) -> str:
    """Horizontal bars, biggest first: for a breakdown by category or payee. The rest are
    combined as "Other".

    Args:
        dataset_id: the id of a breakdown, e.g. "ds-4f2a9c1e".
        title: optional chart title.
        top: how many to show before "Other" (default 8, at most 12).
    """
    data, problem = await _open(ctx, dataset_id)
    if problem:
        return problem
    if data["kind"] != "breakdown":
        return _wrong("bar_chart", data)
    return _breakdown("bar", data, title, max(2, min(int(top), 12)))


@mcp.tool()
async def donut_chart(ctx: Context, dataset_id: str, title: str | None = None) -> str:
    """A ring split into shares of the whole: for a breakdown, when the customer asks for a pie or
    donut. Shows the 5 biggest and "Other" (more slices can't be told apart).

    Args:
        dataset_id: the id of a breakdown, e.g. "ds-4f2a9c1e".
        title: optional chart title.
    """
    data, problem = await _open(ctx, dataset_id)
    if problem:
        return problem
    if data["kind"] != "breakdown":
        return _wrong("donut_chart", data)
    return _breakdown("donut", data, title, 5)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
