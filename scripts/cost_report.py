"""Cost report from the ledger (Postgres or telemetry.db), written by the chat after every turn.

    python -m scripts.cost_report              # last 7 days
    python -m scripts.cost_report --days 30

Sections: totals, per day, per customer, per agent, per model, and guardrails (what fired,
what it cost). Set COST_ALERT_DAILY_USD in .env to get a warning (exit code 2) when a day
goes over budget, e.g. from a scheduled job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

from scripts._common import load_env


def table(title: str, header: tuple, rows: list) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (nothing yet)")
        return
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for row in (header, *rows):
        print("  " + "  ".join(str(x).ljust(w) for x, w in zip(row, widths)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    load_env()
    from observability.costs import Ledger, usd

    ledger = Ledger()
    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat(timespec="seconds")
    q = ledger.query
    turns = q("SELECT total_cost_usd, latency_ms FROM turns WHERE ts >= ?", (since,))
    print(f"Cost report, last {args.days} days  (ledger: {ledger.where})")
    if not turns:
        print("No turns recorded yet. Chat with  python -m scripts.chat_multi  first.")
        return 0
    total = sum(t[0] or 0 for t in turns)
    lat = sorted(t[1] or 0 for t in turns)
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
    print(f"  turns {len(turns)}   total {usd(total)}   per turn {usd(total / len(turns))}   "
          f"p95 latency {p95 / 1000:.1f}s")

    by_day = q("SELECT substr(ts,1,10) d, COUNT(*), SUM(total_cost_usd), SUM(guard_cost_usd) FROM turns "
               "WHERE ts >= ? GROUP BY d ORDER BY d", (since,))
    table("Per day", ("day", "turns", "cost", "of which guardrails"),
          [(d, n, usd(c or 0), usd(g or 0)) for d, n, c, g in by_day])
    table("Per customer", ("customer", "turns", "cost", "per turn"),
          [(c, n, usd(x or 0), usd((x or 0) / n)) for c, n, x in
           q("SELECT customer, COUNT(*), SUM(total_cost_usd) FROM turns WHERE ts >= ? GROUP BY customer", (since,))])
    table("Per agent", ("agent", "model calls", "input tokens", "output tokens", "cost"),
          [(a, n, i, o, usd(c or 0)) for a, n, i, o, c in
           q("SELECT agent, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost_usd) FROM llm_calls "
             "WHERE ts >= ? GROUP BY agent ORDER BY SUM(cost_usd) DESC", (since,))])
    table("Per model", ("model", "calls", "cache read tokens", "cost"),
          [(m, n, cr, usd(c or 0) if c is not None else "no price") for m, n, cr, c in
           q("SELECT model, COUNT(*), SUM(cache_read), SUM(cost_usd) FROM llm_calls WHERE ts >= ? GROUP BY model",
             (since,))])

    checks = q("SELECT layer, source, mode, findings, applied, cost_usd, error FROM guard_checks WHERE ts >= ?", (since,))
    fired = Counter()
    for layer, source, mode, findings, applied, _, error in checks:
        for f in json.loads(findings or "[]"):
            fired[(layer, source, f.split(":")[0] if not f.startswith(("topic:", "pii:")) else f.rsplit(":", 1)[0]
                   if f.startswith("pii:") else f, "acted" if applied else "logged")] += 1
        if error:
            fired[(layer, source, "error", "logged")] += 1
    table("Guardrails", ("layer", "where", "finding", "action", "times"),
          [(*k, n) for k, n in fired.most_common()])
    g_cost = sum(c[5] or 0 for c in checks)
    print(f"  {len(checks)} checks, {usd(g_cost)}")

    limit = os.environ.get("COST_ALERT_DAILY_USD")
    if limit:
        over = [(d, c) for d, _, c, _ in by_day if (c or 0) > float(limit)]
        if over:
            print("\nWARNING: daily budget of " + usd(float(limit)) + " exceeded on "
                  + ", ".join(f"{d} ({usd(c)})" for d, c in over))
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
