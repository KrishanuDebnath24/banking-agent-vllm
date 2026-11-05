"""Seed realistic transactions for customers A and B in the OBP sandbox.

Uses real transfers between each customer's own accounts (from demo_accounts.json):
  income    -> current   : salary / client payments
  current   -> merchants : rent, bills, groceries, transport, dining, shopping
  current   -> savings   : monthly savings transfer

    python -m scripts.seed_transactions                  # dry run: show the plan, change nothing
    python -m scripts.seed_transactions --apply          # create the transactions
    python -m scripts.seed_transactions --apply --customer A --cycles 2

Note: OBP stamps each transaction with the time it is created, so a "cycle"
is one month's worth of activity, not a month in the past.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

from scripts._common import ROOT, load_demo_accounts, login, setup
from obp.client import OBPError

LOG_FILE = ROOT / "seed_log.jsonl"

# Bills and subscriptions keep the same amount every month; card spend varies.
FIXED_KEYWORDS = ("Rent", "Direct Debit", "Mobile", "Broadband", "Membership",
                  "Netflix", "Spotify", "Adobe", "Notion")

# (description, amount) per category. Amounts are in the account currency.
# Every single amount stays below OBP_MAX_TRANSFER (default 800): bigger
# transfers need a second user to approve them. So A is paid weekly.
PROFILES: dict[str, dict] = {
    # Customer A: salaried professional, paid weekly
    "A": {
        "income": [("ACME Analytics Ltd - Weekly pay", 712.50)] * 4,
        "savings": [("Monthly savings transfer", 300.00)],
        "spending": [
            ("Rent - Harbour Lettings", 795.00),
            ("Octopus Energy - Direct Debit", 78.40),
            ("Thames Water - Direct Debit", 32.15),
            ("Vodafone - Mobile", 22.00),
            ("Virgin Media - Broadband", 35.00),
            ("Netflix.com", 10.99),
            ("Spotify UK", 11.99),
            ("PureGym - Membership", 24.99),
            ("Tesco Stores 2041", 54.20),
            ("Tesco Stores 2041", 38.75),
            ("Sainsbury's 0612", 42.30),
            ("Tesco Express 5517", 17.60),
            ("Lidl GB", 27.80),
            ("TfL Travel Charge", 32.40),
            ("Uber Trip", 14.60),
            ("Pret A Manger", 7.45),
            ("Nando's", 28.90),
            ("Dishoom", 46.50),
            ("Amazon.co.uk", 39.99),
            ("Boots", 12.50),
        ],
    },
    # Customer B: freelance designer
    "B": {
        "income": [
            ("Brightwave Studio - Invoice 1042", 780.00),
            ("Northside Media - Invoice 1043", 760.00),
            ("Brightwave Studio - Invoice 1044", 670.00),
        ],
        "savings": [("Tax pot transfer", 400.00)],
        "spending": [
            ("Rent - Camden Homes", 740.00),
            ("EDF Energy - Direct Debit", 64.30),
            ("EE - Mobile", 18.00),
            ("Adobe Creative Cloud", 56.98),
            ("Notion Labs", 8.00),
            ("Netflix.com", 10.99),
            ("Co-op Food", 23.40),
            ("Waitrose", 71.85),
            ("Aldi Stores", 31.20),
            ("Co-op Food", 14.65),
            ("National Rail - Tickets", 48.70),
            ("Lime Bike", 6.25),
            ("Costa Coffee", 4.10),
            ("Costa Coffee", 3.85),
            ("Wagamama", 24.35),
            ("Argos", 44.99),
        ],
    },
}


def build_plan(label: str, accts: dict, cycles: int, rng: random.Random) -> list[dict]:
    """Order per cycle: income first, then savings, then spending shuffled."""
    prof = PROFILES[label]
    plan = []
    for cycle in range(1, cycles + 1):
        def vary(x: float, pct: float) -> float:
            return round(x * (1 + rng.uniform(-pct, pct)), 2)

        for desc, amt in prof["income"]:
            plan.append({"cycle": cycle, "from": "income", "to": "current", "description": desc, "amount": amt})
        for desc, amt in prof["savings"]:
            plan.append({"cycle": cycle, "from": "current", "to": "savings", "description": desc, "amount": amt})
        spending = list(prof["spending"])
        rng.shuffle(spending)
        for desc, amt in spending:
            # fixed bills keep their amount; card spend varies a little
            fixed = amt >= 900 or any(k in desc for k in FIXED_KEYWORDS)
            plan.append({"cycle": cycle, "from": "current", "to": "merchants", "description": desc,
                         "amount": amt if fixed else vary(amt, 0.2)})
    for step in plan:
        step["from_id"] = accts[step["from"]]
        step["to_id"] = accts[step["to"]]
    return plan


def plan_top_ups(client, token, bank_id, accts, plan) -> list[dict]:
    """If the income account can't cover the plan, move money into it from spare
    'funding' accounts first. These transfers land in hidden accounts only."""
    income_bal, _ = client.balance_amount(token, bank_id, accts["income"])
    shortfall = sum(s["amount"] for s in plan if s["from"] == "income") - income_bal
    top_ups = []
    for acc_id in accts.get("funding", []):
        if shortfall <= 0:
            break
        bal, _ = client.balance_amount(token, bank_id, acc_id)
        take = round(min(bal, shortfall + 1), 2)  # +1 leaves a little headroom
        if take < 1:
            continue
        for part in client.split_amount(take):  # stay below the approval threshold
            top_ups.append({"cycle": 0, "from": "funding", "to": "income", "description": "Internal transfer",
                            "amount": part, "from_id": acc_id, "to_id": accts["income"]})
        shortfall -= take
    return top_ups


def too_big(plan: list[dict], limit: float) -> list[str]:
    """Visible transfers can't be split without looking odd, so flag them instead."""
    return [f"{s['description']} {s['amount']:.2f}" for s in plan if s["amount"] > limit]


def check_funds(client, token, bank_id, accts, plan) -> list[str]:
    """Warn if income or current would run out part-way (top-ups included)."""
    warnings = []
    income_bal, _ = client.balance_amount(token, bank_id, accts["income"])
    current_bal, _ = client.balance_amount(token, bank_id, accts["current"])
    income_in = sum(s["amount"] for s in plan if s["to"] == "income")
    income_out = sum(s["amount"] for s in plan if s["from"] == "income")
    current_in = sum(s["amount"] for s in plan if s["to"] == "current")
    current_out = sum(s["amount"] for s in plan if s["from"] == "current")
    if income_out > income_bal + income_in:
        warnings.append(f"income account has {income_bal + income_in:.2f} (with top-ups) "
                        f"but the plan pays out {income_out:.2f}; try fewer cycles")
    if current_out > current_bal + current_in:
        warnings.append(f"current account would go below zero ({current_bal + current_in - current_out:.2f})")
    return warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="create the transactions (default is a dry run)")
    ap.add_argument("--customer", choices=["A", "B"], help="only seed one customer")
    ap.add_argument("--cycles", type=int, default=1, help="months of activity to create (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for amount variation")
    args = ap.parse_args()

    client, _ = setup()
    demo = load_demo_accounts()
    bank_id = demo["bank_id"]
    labels = [args.customer] if args.customer else list(demo["customers"])
    rng = random.Random(args.seed)

    for label in labels:
        accts = demo["customers"][label]
        if accts.get("username") != os.environ.get(f"CUSTOMER_{label}_USERNAME"):
            print(f"\nCustomer {label} in .env differs from demo_accounts.json. "
                  "Run: python -m scripts.list_accounts --write --replace")
            continue
        currency = accts["currency"]
        user, token = login(client, label)
        plan = build_plan(label, accts, args.cycles, rng)
        plan = plan_top_ups(client, token, bank_id, accts, plan) + plan

        print(f"\n=== Customer {label}: {user} ({len(plan)} transfers, {currency}) ===")
        for s in plan:
            tag = "top-up" if s["cycle"] == 0 else f"c{s['cycle']}"
            print(f"  {tag:<6} {s['from']:>7} -> {s['to']:<9} {s['amount']:>9.2f}  {s['description']}")

        warnings = check_funds(client, token, bank_id, accts, plan)
        warnings += [f"'{x}' is above OBP_MAX_TRANSFER {client.max_transfer:.2f}; lower it in PROFILES"
                     for x in too_big(plan, client.max_transfer)]
        for w in warnings:
            print(f"  WARNING: {w}")

        if not args.apply:
            print("  Dry run. Add --apply to create these transactions.")
            continue
        if warnings:
            print("  Skipping this customer: fix the warnings (pick a richer income account or use fewer cycles).")
            continue

        ok = failed = 0
        with LOG_FILE.open("a") as log:
            for s in plan:
                entry = {"at": datetime.now(timezone.utc).isoformat(), "customer": label, **s}
                try:
                    res = client.transfer(token, bank_id, s["from_id"], s["to_id"],
                                          s["amount"], currency, s["description"])
                    entry.update(status=res.get("status"), transaction_request_id=res.get("id"))
                    ok += 1
                    print(f"  OK    {s['amount']:>9.2f}  {s['description']}  [{res.get('status')}]")
                except OBPError as e:
                    entry.update(status="ERROR", error=e.body[:300])
                    failed += 1
                    print(f"  FAIL  {s['amount']:>9.2f}  {s['description']}  -> HTTP {e.status}: {e.body[:160]}")
                    if failed == 1 and ok == 0:
                        log.write(json.dumps(entry) + "\n")
                        print("  First transfer failed; stopping so you can check the error.")
                        return 1
                log.write(json.dumps(entry) + "\n")
                time.sleep(0.5)  # be gentle with the shared sandbox
        print(f"  Done: {ok} created, {failed} failed. Log: {LOG_FILE.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
