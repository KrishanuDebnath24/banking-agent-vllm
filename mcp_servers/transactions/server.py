"""Transactions MCP server: history, spending totals, month-by-month totals, breakdowns, statements.

All totals are computed here in code, so answers never depend on the model's arithmetic.

- Categories (groceries, eating out, ...) are looked up in obp/categories.py: the AI names the
  category and this server knows its payee words.
- Totals leave out transfers between the customer's own accounts: money moved from current to
  savings is neither spending nor income. Lists and statements still show them.
- Month-by-month totals, day-by-day totals, month-to-date comparisons (running totals of this
  month and the same days of last month) and breakdowns are saved for charts
  (mcp_servers/datasets.py) and come back with a dataset id; the chart specialist draws them from
  that id. Differences and percentages are worked out here too, never by the model.
"""

from __future__ import annotations

import functools
import json
from collections import defaultdict
from datetime import date, timedelta

from mcp.server.fastmcp import Context, FastMCP

from mcp_servers import datasets
from mcp_servers.common import (PORTS, OBPError, bank, customer, error, parse_date, rows_for,
                                 transport_security)
from obp import categories
from obp.search import payee_matches

mcp = FastMCP("transactions", port=PORTS["transactions"], stateless_http=True, json_response=True,
              transport_security=transport_security(),
              instructions="Transaction history, spending totals, monthly figures and statements for the "
                           "logged-in customer.")

MAX_MONTHS = 24
MAX_SERIES = 4
CHART_HINT = "To chart these figures, give the chart specialist this dataset_id (not the figures)."


def _dates(from_date: str | None, to_date: str | None) -> tuple[str | None, str | None]:
    f, t = parse_date(from_date, "from_date"), parse_date(to_date, "to_date")
    if f and t and f > t:
        raise ValueError(f"from_date ({f}) is after to_date ({t}): the dates are the wrong way round")
    return f, t


def _requested(from_date, to_date) -> dict:
    """The period the caller asked for, so the answer can state it (not guess it from the data)."""
    return {"from": parse_date(from_date, "from_date") or "earliest available",
            "to": parse_date(to_date, "to_date") or "latest available"}


def _period(rows_by_acc: dict) -> dict:
    days = [r["date"] for rows in rows_by_acc.values() for r in rows if r["date"]]
    return {"first_transaction": min(days) if days else None, "last_transaction": max(days) if days else None}


def _any_account(account: str | None) -> str | None:
    """None for "all of them". Models often send "" or "all" instead of leaving the account out;
    that must not look like someone else's account."""
    if account is None or account.strip().lower() in ("", "all", "both", "any", "all accounts", "none", "null"):
        return None
    return account


async def _rows(ctx, account, search, from_date, to_date):
    """Rows per account, as the bank lists them. account=None means all of the customer's accounts."""
    cust = await customer(ctx)
    account = _any_account(account)
    if account is None:
        ids = list(cust.accounts.values())
    else:
        acc_id = cust.resolve(account)
        if acc_id is None:
            return cust, None
        ids = [acc_id]
    f, t = _dates(from_date, to_date)
    out = {}
    for acc_id in ids:
        raw = await bank(cust.obp.transactions, cust.token, cust.bank_id, acc_id)
        out[cust.name_of(acc_id)] = rows_for(cust, raw, f, t, search)
    return cust, out


def _mark_own_transfers(rows: list[dict]) -> None:
    """A transfer between the customer's own accounts shows up twice: the same amount out of one
    account and into the other on the same day. Both halves are marked, so totals can leave them out."""
    arrivals: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["amount"] > 0:
            arrivals[(r["date"], round(r["amount"], 2))].append(r)
    for r in rows:
        if r["amount"] < 0:
            for other in arrivals.get((r["date"], round(-r["amount"], 2)), []):
                if other["account"] != r["account"] and not other.get("own_transfer"):
                    r["own_transfer"] = other["own_transfer"] = True
                    break
    for r in rows:
        r.setdefault("own_transfer", False)


async def _everything(ctx, account):
    """(customer, rows, chosen account): all the customer's transactions, each with its account and
    whether it's half of an own transfer (worked out across both accounts, even when one is chosen).
    rows is None if the account isn't the customer's."""
    cust = await customer(ctx)
    chosen = None
    account = _any_account(account)
    if account is not None:
        acc_id = cust.resolve(account)
        if acc_id is None:
            return cust, None, None
        chosen = cust.name_of(acc_id)
    rows = []
    for acc_id in cust.accounts.values():
        raw = await bank(cust.obp.transactions, cust.token, cust.bank_id, acc_id)
        rows += [{**r, "account": cust.name_of(acc_id)} for r in rows_for(cust, raw, None, None, None)]
    _mark_own_transfers(rows)
    if chosen:
        rows = [r for r in rows if r["account"] == chosen]
    return cust, rows, chosen


def _groups(category: str | None, search: str | None) -> list[tuple[str | None, tuple[str, ...]]]:
    """What to count: [(category name, payee words)]; one entry per category, or (None, words) for
    a search, or (None, ()) for everything."""
    if category and search:
        raise ValueError("give either category or search, not both")
    if category:
        names, table = categories.words_for(category)
        return [(n, table[n]) for n in names]
    if search:
        return [(None, tuple(_Term(w.strip().lower()) for w in search.split(",") if w.strip()))]
    return [(None, ())]


class _Term(str):
    """A payee the customer named (search): matched forgivingly ("Leap Card top-ups" finds "Leap Card
    top-up"). Category words are the bank's own and stay exact."""


def _hit(description: str, words: tuple[str, ...]) -> bool:
    return not words or any(payee_matches(description, w) if isinstance(w, _Term) else w in description.lower()
                            for w in words)


def _direction(r: dict, money: str) -> bool:
    return r["amount"] < 0 if money == "out" else r["amount"] > 0


def _filter_info(category, search) -> dict | str:
    if category:
        return {"categories": categories.words_for(category)[0]}
    return {"search": search} if search else "everything"


def _left_out(rows: list[dict]) -> dict | None:
    moved = [r for r in rows if r["own_transfer"]]
    if not moved:
        return None
    return {"count": len(moved), "amount": round(sum(abs(r["amount"]) for r in moved), 2),
            "why": "transfers between the customer's own accounts are neither spending nor income"}


def _errors(fn):
    """The same plain errors from every tool."""
    @functools.wraps(fn)
    async def wrapped(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ValueError as e:
            return error("invalid_input", message=str(e))
        except PermissionError as e:
            return error("not_authorised", detail=str(e))
        except OBPError as e:
            return error("bank_error", status=e.status)
    return wrapped


@mcp.tool()
@_errors
async def get_transactions(ctx: Context, account: str | None = None, search: str | None = None,
                           from_date: str | None = None, to_date: str | None = None, limit: int = 30) -> str:
    """List transactions, newest first. Negative amounts are money out, positive are money in.

    Args:
        account: "current" or "savings"; leave empty for all accounts.
        search: optional comma-separated words to match in the description, e.g. "tesco,lidl".
        from_date: optional first day, YYYY-MM-DD.
        to_date: optional last day, YYYY-MM-DD.
        limit: maximum transactions per account (default 30, max 100).
    """
    cust, all_acc = await _rows(ctx, None, search, from_date, to_date)  # both accounts: own transfers
    chosen = _any_account(account)
    if chosen is not None:
        acc_id = cust.resolve(chosen)
        if acc_id is None:
            return cust.denied()
        chosen = cust.name_of(acc_id)
    flat = [{**r, "account": a} for a, rows in all_acc.items() for r in rows]
    _mark_own_transfers(flat)
    by_acc: dict[str, list[dict]] = {a: [] for a in all_acc if chosen is None or a == chosen}
    for r in flat:
        if r["account"] in by_acc:
            row = {k: v for k, v in r.items() if k not in ("account", "own_transfer")}
            if r["own_transfer"]:
                row["own_transfer"] = True  # half of a move between the customer's own accounts
            by_acc[r["account"]].append(row)
    counted = [r for rows in by_acc.values() for r in rows if not r.get("own_transfer")]
    limit = max(1, min(int(limit), 100))
    return json.dumps({
        "currency": cust.currency, "requested_period": _requested(from_date, to_date),
        "transactions_found_between": _period(by_acc),
        "accounts": {a: {"count": len(r), "transactions": r[:limit]} for a, r in by_acc.items()},
        # Worked out here so no model adds up a list (it got income wrong by counting own transfers).
        "totals": {"money_in": round(sum(r["amount"] for r in counted if r["amount"] > 0), 2),
                   "money_out": round(-sum(r["amount"] for r in counted if r["amount"] < 0), 2),
                   "about": "all transactions found, not only those listed, leaving out moves between the "
                            "customer's own accounts (own_transfer); give these totals, don't add amounts up"}})


@mcp.tool()
@_errors
async def summarize_spending(ctx: Context, account: str | None = None, category: str | None = None,
                             search: str | None = None, from_date: str | None = None,
                             to_date: str | None = None) -> str:
    """Totals for one or all accounts: money in, money out, spending per payee and income per payer,
    plus the date range covered. Use for any "how much did I spend / earn" question. Transfers
    between the customer's own accounts are left out of the totals.

    Args:
        account: "current" or "savings"; leave empty for all accounts in one call.
        category: a spending or income category, e.g. "groceries" or "eating out" (the bank knows
            which payees are in it); comma-separated for several together.
        search: instead of a category, payee words, e.g. "tesco"; comma-separated for several
            payees in one total, e.g. "netflix,spotify".
        from_date: optional first day, YYYY-MM-DD.
        to_date: optional last day, YYYY-MM-DD.
    """
    f, t = _dates(from_date, to_date)
    words = tuple(w for _, ws in _groups(category, search) for w in ws)
    cust, rows, chosen = await _everything(ctx, account)
    if rows is None:
        return cust.denied()
    rows = [r for r in rows if (not f or r["date"] >= f) and (not t or r["date"] <= t)
            and _hit(r["description"], words)]
    names = [chosen] if chosen else list(cust.accounts)
    accounts, total_in, total_out = {}, 0.0, 0.0
    for name in names:
        mine = [r for r in rows if r["account"] == name and not r["own_transfer"]]
        spend, income = defaultdict(float), defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for r in mine:
            (spend if r["amount"] < 0 else income)[r["description"]] += abs(r["amount"])
            counts[r["description"]] += 1
        m_in, m_out = round(sum(income.values()), 2), round(sum(spend.values()), 2)
        total_in += m_in
        total_out += m_out
        accounts[name] = {
            "transactions": len(mine), "money_in": m_in, "money_out": m_out,
            "spending_by_payee": [{"payee": p, "total": round(v, 2), "count": counts[p]}
                                  for p, v in sorted(spend.items(), key=lambda kv: -kv[1])],
            "income_by_payer": [{"payer": p, "total": round(v, 2), "count": counts[p]}
                                for p, v in sorted(income.items(), key=lambda kv: -kv[1])],
        }
    result = {"currency": cust.currency, "filter": _filter_info(category, search),
              "requested_period": _requested(from_date, to_date),
              "transactions_found_between": _period({"all": rows}),
              "accounts": accounts,
              "totals": {"money_in": round(total_in, 2), "money_out": round(total_out, 2)}}
    if left := _left_out(rows):
        result["own_transfers_left_out"] = left
    return json.dumps(result)


def _month_starts(start: date, end: date) -> list[date]:
    out, day = [], start.replace(day=1)
    while day <= end:
        out.append(day)
        day = (day + timedelta(days=32)).replace(day=1)
    return out


def _last_day(month: date) -> date:
    return (month + timedelta(days=32)).replace(day=1) - timedelta(days=1)


def _changes(keys: list[str], by_month: list[float]) -> list[dict]:
    """How each month compares with the one before, worked out here so the AI never subtracts."""
    out = []
    for i in range(1, len(keys)):
        before, now = by_month[i - 1], by_month[i]
        diff = round(now - before, 2)
        out.append({"month": keys[i], "compared_with": keys[i - 1], "difference": abs(diff),
                    "more_or_less": "more" if diff > 0 else "less" if diff < 0 else "the same",
                    "percent": round(abs(diff) / before * 100, 1) if before else None})
    return out


@mcp.tool()
@_errors
async def spending_by_month(ctx: Context, category: str | None = None, search: str | None = None,
                            from_date: str | None = None, to_date: str | None = None,
                            account: str | None = None, money: str = "out", label: str | None = None) -> str:
    """Month-by-month totals in ONE call, saved for a chart: use for "each month", "monthly",
    trends, comparing months ("how much more than August?"), and any chart of spending or income
    over time. Never call summarize_spending once per month instead. Each line of figures includes
    change_from_previous_month (difference and percent, worked out here: quote it, don't subtract).
    Returns a dataset_id for the chart specialist.

    Args:
        category: a category, e.g. "groceries"; up to 4 comma-separated, e.g. "groceries,eating out",
            for one line of figures each (to compare them). Leave empty, with no search, for all
            spending (or all income).
        search: instead of a category, payee words for one particular payee, e.g. "tesco".
        from_date: optional first day, YYYY-MM-DD ("this year" = January 1st).
        to_date: optional last day, YYYY-MM-DD.
        account: "current" or "savings"; leave empty for all accounts.
        money: "out" for spending (default) or "in" for income.
        label: a short title for the chart, e.g. "Groceries".
    """
    if money not in ("out", "in"):
        raise ValueError('money must be "out" or "in"')
    f, t = _dates(from_date, to_date)
    groups = _groups(category, search)
    if len(groups) > MAX_SERIES:
        raise ValueError(f"at most {MAX_SERIES} categories at once")
    cust, rows, _ = await _everything(ctx, account)
    if rows is None:
        return cust.denied()
    dated = [r for r in rows if r["date"]]
    today = date.today()
    records_start = min((r["date"] for r in dated), default=None)
    start = date.fromisoformat(records_start) if records_start else None
    if start and f:
        start = max(start, date.fromisoformat(f))
    end = min(date.fromisoformat(t), today) if t else today
    everything = "Spending" if money == "out" else "Income"
    title = " ".join((label or "").split())[:40] or (
        " and ".join(n.capitalize() for n, _ in groups) if category else
        (search.split(",")[0].strip().title() if search else everything))
    base = {"currency": cust.currency, "label": title, "money": money, "filter": _filter_info(category, search),
            "requested_period": _requested(from_date, to_date), "records_start": records_start}
    if start is None or start > end:
        return json.dumps({**base, "months": [], "series": [], "total": 0.0, "note": (
            "No transactions in the requested period." if records_start is None else
            f"The records here start on {records_start}, after the requested period.")})

    in_range = [r for r in dated if start.isoformat() <= r["date"] <= end.isoformat() and _direction(r, money)]
    months = _month_starts(start, end)
    dropped = max(0, len(months) - MAX_MONTHS)
    months = months[dropped:]
    first_day = max(start, months[0])
    keys = [m.strftime("%Y-%m") for m in months]
    partial = {}
    for m, key in zip(months, keys):
        text = []
        if m == months[0] and first_day.day != 1:
            text.append(f"from {first_day.isoformat()}")
        if m == months[-1] and (end < _last_day(m) or end == today):  # today isn't over, even on the 31st
            text.append(f"to {end.isoformat()}" + (" (so far)" if end == today else ""))
        if text:
            partial[key] = " ".join(text)

    series, counted = [], []
    for name, words in groups:
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for r in in_range:
            if _hit(r["description"], words) and r["date"][:7] in keys:
                counted.append(r)
                if not r["own_transfer"]:
                    totals[r["date"][:7]] += abs(r["amount"])
                    counts[r["date"][:7]] += 1
        by_month = [round(totals[k], 2) for k in keys]
        series.append({"name": name.capitalize() if name else title, "total": round(sum(by_month), 2),
                       "by_month": by_month, "payments": [counts[k] for k in keys]})

    note = None
    if f and records_start > f:
        note = f"The records here start on {records_start}; there's nothing earlier to show."
    if dropped:
        note = f"Showing the latest {MAX_MONTHS} months."
    payload = {"label": title, "money": money, "currency": cust.currency, "months": keys, "partial": partial,
               "series": series, "period": {"from": first_day.isoformat(), "to": end.isoformat()},
               "records_start": records_start, "note": note}
    # The change from month to month goes to the AI only; the saved chart figures stay as they were.
    compared = [{**s, "change_from_previous_month": _changes(keys, s["by_month"])} for s in series]
    result = {**base, "period": payload["period"], "months": keys, "partial": partial, "series": compared,
              "total": round(sum(s["total"] for s in series), 2)}
    if keys[-1] in partial and end == today:
        result["this_month_so_far"] = (f"{keys[-1]} is only up to today ({end.isoformat()}), not a full month: "
                                       "say so, so a low figure isn't taken for a drop")
    if note:
        result["note"] = note
    if left := _left_out(counted):
        result["own_transfers_left_out"] = left
    result["dataset_id"] = datasets.save(cust.records_key, "monthly", payload)
    result["chart"] = CHART_HINT
    return json.dumps(result)


MAX_DAYS = 62


def _one_group(category: str | None, search: str | None) -> tuple[str | None, tuple[str, ...]]:
    groups = _groups(category, search)
    if len(groups) > 1:
        raise ValueError("one category at a time for daily figures")
    return groups[0]


def _label(label: str | None, name: str | None, search: str | None, money: str) -> str:
    return " ".join((label or "").split())[:40] or (
        name.capitalize() if name else search.split(",")[0].strip().title() if search
        else "Spending" if money == "out" else "Income")


def _money_text(value: float, currency: str) -> str:
    return f"£{value:,.2f}" if currency == "GBP" else f"{value:,.2f} {currency}"


def _day_text(d: date) -> str:
    return f"{d.day} {d.strftime('%b %Y')}"


def _daily_totals(rows: list[dict], start: date, end: date, words: tuple[str, ...],
                  money: str) -> tuple[dict, dict, list[dict]]:
    """(total per date, payments per date, the rows counted) between start and end, own transfers
    left out of the totals."""
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    counted = []
    lo, hi = start.isoformat(), end.isoformat()
    for r in rows:
        if r["date"] and lo <= r["date"] <= hi and _direction(r, money) and _hit(r["description"], words):
            counted.append(r)
            if not r["own_transfer"]:
                totals[r["date"]] += abs(r["amount"])
                counts[r["date"]] += 1
    return totals, counts, counted


@mcp.tool()
@_errors
async def spending_by_day(ctx: Context, category: str | None = None, search: str | None = None,
                          from_date: str | None = None, to_date: str | None = None,
                          account: str | None = None, money: str = "out", label: str | None = None) -> str:
    """Day-by-day totals for up to 62 days (default: this month so far), saved for a chart: use for
    "each day", "daily", "which days", or a chart of a month day by day. Days with nothing count as
    0. The highest day and the total are worked out here. Returns a dataset_id for the chart
    specialist.

    Args:
        category: one category, e.g. "groceries"; leave empty, with no search, for all spending.
        search: instead of a category, payee words, e.g. "tesco".
        from_date: first day, YYYY-MM-DD (default: the 1st of this month).
        to_date: last day, YYYY-MM-DD (default: today).
        account: "current" or "savings"; leave empty for all accounts.
        money: "out" for spending (default) or "in" for income.
        label: a short title for the chart, e.g. "Groceries".
    """
    if money not in ("out", "in"):
        raise ValueError('money must be "out" or "in"')
    f, t = _dates(from_date, to_date)
    name, words = _one_group(category, search)
    today = date.today()
    start = date.fromisoformat(f) if f else today.replace(day=1)
    end = min(date.fromisoformat(t), today) if t else today
    if (end - start).days + 1 > MAX_DAYS:
        raise ValueError(f"at most {MAX_DAYS} days at once; for longer periods use spending_by_month")
    cust, rows, _ = await _everything(ctx, account)
    if rows is None:
        return cust.denied()
    title = _label(label, name, search, money)
    records_start = min((r["date"] for r in rows if r["date"]), default=None)
    first = max(start, date.fromisoformat(records_start)) if records_start else start
    base = {"currency": cust.currency, "label": title, "money": money, "filter": _filter_info(category, search),
            "requested_period": {"from": start.isoformat(), "to": (t or end.isoformat())},
            "records_start": records_start}
    if first > end:
        return json.dumps({**base, "days": [], "total": 0.0, "note": (
            "Nothing yet: that period hasn't started." if start > today else
            "No transactions in the requested period." if records_start is None else
            f"The records here start on {records_start}, after the requested period.")})

    keys = [(first + timedelta(days=i)).isoformat() for i in range((end - first).days + 1)]
    totals, counts, counted = _daily_totals(rows, first, end, words, money)
    by_day = [round(totals[k], 2) for k in keys]
    total = round(sum(by_day), 2)
    top = max(range(len(keys)), key=lambda i: by_day[i])
    notes = []
    if first > start:
        notes.append(f"The records here start on {records_start}; there's nothing earlier to show.")
    if end == today:
        notes.append(f"Up to today ({today.isoformat()}), so today is only so far.")
    note = " ".join(notes) or None
    payload = {"label": f"{title} by day", "money": money, "currency": cust.currency, "days": keys,
               "series": [{"name": title, "by_day": by_day, "total": total, "payments": [counts[k] for k in keys]}],
               "period": {"from": keys[0], "to": keys[-1]}, "records_start": records_start, "note": note}
    result = {**base, "period": payload["period"], "days": len(keys), "total": total,
              "days_with_any": sum(1 for v in by_day if v),
              "average_per_day": round(total / len(keys), 2),
              "highest_day": {"date": keys[top], "amount": by_day[top]} if total else None,
              # only the days with something, to keep this short; the chart has every day
              "by_day": [{"date": k, "amount": v, "payments": counts[k]} for k, v in zip(keys, by_day) if v]}
    if note:
        result["note"] = note
    if left := _left_out(counted):
        result["own_transfers_left_out"] = left
    result["dataset_id"] = datasets.save(cust.records_key, "daily", payload)
    result["chart"] = CHART_HINT
    return json.dumps(result)


@mcp.tool()
@_errors
async def month_to_date(ctx: Context, category: str | None = None, search: str | None = None,
                        as_of: str | None = None, account: str | None = None, money: str = "out",
                        label: str | None = None) -> str:
    """This month so far against the same days of last month, worked out here: e.g. 1-16 October
    against 1-16 September, with the difference and percent (quote them exactly; never work them
    out). Use for "so far this month compared with last month", "am I spending more than last month
    at this point", "up to the 16th". Also saves both months' running totals for a line chart and
    returns a dataset_id.

    Args:
        category: one category, e.g. "groceries"; leave empty, with no search, for all spending.
        search: instead of a category, payee words, e.g. "tesco".
        as_of: the last day to count, YYYY-MM-DD (default: today). Its month, from the 1st, is
            compared with the month before, from the 1st to the same day of the month.
        account: "current" or "savings"; leave empty for all accounts.
        money: "out" for spending (default) or "in" for income.
        label: a short title for the chart, e.g. "Groceries".
    """
    if money not in ("out", "in"):
        raise ValueError('money must be "out" or "in"')
    name, words = _one_group(category, search)
    today = date.today()
    end = date.fromisoformat(parse_date(as_of, "as_of")) if as_of else today
    if end > today:
        raise ValueError(f"as_of ({end.isoformat()}) is in the future; the latest day is today ({today.isoformat()})")
    this_start = end.replace(day=1)
    prev_last = this_start - timedelta(days=1)
    prev_start = prev_last.replace(day=1)
    prev_end = prev_start.replace(day=min(end.day, prev_last.day))
    cust, rows, _ = await _everything(ctx, account)
    if rows is None:
        return cust.denied()
    title = _label(label, name, search, money)
    cur = cust.currency
    records_start = min((r["date"] for r in rows if r["date"]), default=None)

    def window(a: date, b: date) -> tuple[float, int, list[float], list[dict], list[float], list[int]]:
        """(total, payments, running total by day, rows counted, amount each day, payments each day)."""
        totals, counts, counted = _daily_totals(rows, a, b, words, money)
        running, daily, paid, so_far = [], [], [], 0.0
        for day in range(1, b.day + 1):
            key = a.replace(day=day).isoformat()
            so_far += totals.get(key, 0.0)
            running.append(round(so_far, 2))
            daily.append(round(totals.get(key, 0.0), 2))
            paid.append(counts.get(key, 0))
        return round(so_far, 2), sum(counts.values()), running, counted, daily, paid

    now_total, now_count, now_running, counted_now, now_daily, now_paid = window(this_start, end)
    then_total, then_count, then_running, counted_then, _, _ = window(prev_start, prev_end)
    diff = round(now_total - then_total, 2)
    more_or_less = "more" if diff > 0 else "less" if diff < 0 else "the same"
    percent = round(abs(diff) / then_total * 100, 1) if then_total else None
    this_name, prev_name = this_start.strftime("%B %Y"), prev_start.strftime("%B %Y")
    notes = []
    if records_start and records_start > prev_start.isoformat():
        notes.append(f"The records here start on {records_start}, so {prev_name}'s figure only counts from then "
                     "and the comparison isn't like for like: say so.")
    if prev_end.day < end.day:
        notes.append(f"{prev_name} has only {prev_last.day} days, so its figure is the whole month.")
    if end == today:
        notes.append(f"{this_name} is only up to today ({today.isoformat()}).")
    change = {"difference": abs(diff), "more_or_less": more_or_less, "percent": percent}
    sentence = (f"{title} {_day_text(this_start)} to {_day_text(end)}: {_money_text(now_total, cur)} "
                f"({now_count} payment{'s' if now_count != 1 else ''}). Same days of {prev_name}, "
                f"{_day_text(prev_start)} to {_day_text(prev_end)}: {_money_text(then_total, cur)} "
                f"({then_count} payment{'s' if then_count != 1 else ''}). ")
    sentence += ("The same as last month." if diff == 0 else
                 f"That's {_money_text(abs(diff), cur)} {more_or_less}"
                 + (f" ({percent}% {more_or_less})." if percent is not None else
                    f" (no percentage: nothing in {prev_name} to compare with)."))
    labels = [str(d) for d in range(1, end.day + 1)]
    then_line = (then_running + [then_running[-1]] * (len(labels) - len(then_running)))[:len(labels)] \
        if then_running else [0.0] * len(labels)
    payload = {"label": f"{title}: {this_start.strftime('%b')} so far against {prev_start.strftime('%b')}",
               "money": money, "currency": cur, "days": labels,
               "series": [{"name": this_name, "running": now_running, "total": now_total,
                           "daily": now_daily, "payments": now_paid},   # for the combined chart's columns
                          {"name": prev_name, "running": then_line, "total": then_total}],
               "period": {"from": prev_start.isoformat(), "to": end.isoformat()}, "as_of": end.isoformat(),
               "note": " ".join(n for n in notes if "say so" not in n) or None}
    result = {"currency": cur, "label": title, "money": money, "filter": _filter_info(category, search),
              "this_month": {"from": this_start.isoformat(), "to": end.isoformat(), "total": now_total,
                             "payments": now_count},
              "same_days_last_month": {"from": prev_start.isoformat(), "to": prev_end.isoformat(),
                                       "total": then_total, "payments": then_count},
              "change": change, "in_words": sentence, "records_start": records_start}
    if notes:
        result["note"] = " ".join(notes)
    if left := _left_out(counted_now + counted_then):
        result["own_transfers_left_out"] = left
    result["dataset_id"] = datasets.save(cust.records_key, "pace", payload)
    result["chart"] = (CHART_HINT + " It draws as a line chart of both months' running totals, or as a "
                       "combined chart: a column for each day of this month, with the running totals as lines.")
    return json.dumps(result)


@mcp.tool()
@_errors
async def spending_breakdown(ctx: Context, by: str = "category", category: str | None = None,
                             search: str | None = None, from_date: str | None = None,
                             to_date: str | None = None, account: str | None = None, money: str = "out",
                             label: str | None = None) -> str:
    """Totals split by category or by payee for one period, biggest first, saved for a chart: use
    for "where does my money go", "spending by category", "which shops", and pie/donut or bar
    charts of a breakdown. Returns a dataset_id for the chart specialist.

    Args:
        by: "category" (groceries, bills, ...) or "payee" (each shop or payer).
        category: with by="payee", only these categories' payees, e.g. "groceries" for "which
            supermarkets".
        search: with by="payee", only payees with these words.
        from_date: optional first day, YYYY-MM-DD.
        to_date: optional last day, YYYY-MM-DD.
        account: "current" or "savings"; leave empty for all accounts.
        money: "out" for spending (default) or "in" for income.
        label: a short title for the chart.
    """
    if money not in ("out", "in"):
        raise ValueError('money must be "out" or "in"')
    if by not in ("category", "payee"):
        raise ValueError('by must be "category" or "payee"')
    if by == "category" and (category or search):
        raise ValueError('by="category" already covers every category; to split one category by shop, '
                         'use by="payee" with that category')
    f, t = _dates(from_date, to_date)
    words = tuple(w for _, ws in _groups(category, search) for w in ws)
    cust, rows, _ = await _everything(ctx, account)
    if rows is None:
        return cust.denied()
    dated = [r for r in rows if r["date"]]
    records_start = min((r["date"] for r in dated), default=None)
    first = max(filter(None, [f, records_start]), default=None)
    last = min(t, date.today().isoformat()) if t else date.today().isoformat()
    picked = [r for r in dated if (not first or r["date"] >= first) and r["date"] <= last
              and _direction(r, money) and _hit(r["description"], words)]
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in picked:
        if r["own_transfer"]:
            continue
        name = (categories.category_of(r["description"], money_in=money == "in") or "other").capitalize() \
            if by == "category" else r["description"][:60]
        totals[name] += abs(r["amount"])
        counts[name] += 1
    items = [{"name": n, "total": round(v, 2), "payments": counts[n]}
             for n, v in sorted(totals.items(), key=lambda kv: -kv[1])]
    if len(items) > 50:   # keep the 50 biggest; the rest together
        rest = items[49:]
        items = items[:49] + [{"name": "Everything else", "total": round(sum(i["total"] for i in rest), 2),
                               "payments": sum(i["payments"] for i in rest)}]
    what = "Spending" if money == "out" else "Income"
    title = " ".join((label or "").split())[:40] or (
        f"{what} by category" if by == "category" else
        f"{category.split(',')[0].strip().capitalize() if category else what} by {'shop' if money == 'out' else 'payer'}")
    period = {"from": first or "no transactions", "to": last}
    payload = {"label": title, "money": money, "currency": cust.currency, "by": by, "items": items,
               "total": round(sum(i["total"] for i in items), 2), "period": period}
    result = {"currency": cust.currency, "label": title, "money": money, "by": by,
              "filter": _filter_info(category, search), "requested_period": _requested(from_date, to_date),
              "period": period, "items": items, "total": payload["total"]}
    if f and records_start and records_start > f:
        result["note"] = f"The records here start on {records_start}; there's nothing earlier to show."
    if left := _left_out(picked):
        result["own_transfers_left_out"] = left
    if items:
        result["dataset_id"] = datasets.save(cust.records_key, "breakdown", payload)
        result["chart"] = CHART_HINT
    return json.dumps(result)


@mcp.tool()
@_errors
async def get_statement(account: str, ctx: Context, from_date: str | None = None, to_date: str | None = None) -> str:
    """A statement for one account: opening balance, each transaction in date order with the
    running balance, and closing balance.

    Args:
        account: "current" or "savings".
        from_date: optional first day, YYYY-MM-DD.
        to_date: optional last day, YYYY-MM-DD.
    """
    cust, by_acc = await _rows(ctx, account, None, from_date, to_date)
    if by_acc is None:
        return cust.denied()
    name, rows = next(iter(by_acc.items()))
    lines = list(reversed(rows))  # oldest first
    closing = float(lines[-1]["balance_after"]) if lines and lines[-1]["balance_after"] is not None else None
    opening = round(closing - sum(r["amount"] for r in lines), 2) if closing is not None else None
    found = _period(by_acc)
    period = {"from": parse_date(from_date, "from_date") or found["first_transaction"],
              "to": parse_date(to_date, "to_date") or found["last_transaction"]}
    return json.dumps({"account": name, "currency": cust.currency, "statement_period": period,
                       "transactions_found_between": found,
                       "opening_balance": opening, "opening_balance_date": period["from"],
                       "closing_balance": closing, "closing_balance_date": period["to"],
                       "money_in": round(sum(r["amount"] for r in lines if r["amount"] > 0), 2),
                       "money_out": round(-sum(r["amount"] for r in lines if r["amount"] < 0), 2),
                       "lines": lines})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
