"""List each customer's accounts with balance and transaction count,
and suggest which ones to use for the demo.

    python -m scripts.list_accounts           # show the table and a suggestion
    python -m scripts.list_accounts --write   # also save the suggestion to demo_accounts.json

Roles in demo_accounts.json:
  current   - everyday account the agent shows (salary in, spending out)
  savings   - second account the agent shows (monthly transfer in)
  income    - HIDDEN: plays the employer / clients that pay the customer
  merchants - HIDDEN: plays the shops and billers the customer pays
"""

from __future__ import annotations

import argparse
import json
import time

from scripts._common import DEMO_ACCOUNTS_FILE, login, setup
from obp.client import OBPError


def survey(client, bank_id: str, token: str) -> list[dict]:
    rows = []
    for acc in client.accounts_held(token, bank_id):
        acc_id = acc["id"]
        try:
            amount, currency = client.balance_amount(token, bank_id, acc_id)
        except OBPError:
            amount, currency = 0.0, "?"
        try:
            n_tx = len(client.transactions(token, bank_id, acc_id))
        except OBPError:
            n_tx = -1  # no owner view on this account
        rows.append({"id": acc_id, "label": acc.get("label", ""), "balance": amount,
                     "currency": currency, "tx": n_tx})
        time.sleep(0.2)  # be gentle with the shared sandbox
    return rows


def suggest(rows: list[dict]) -> dict | None:
    """Richest account pays income; then pick current, savings, merchants from the rest."""
    usable = [r for r in rows if r["tx"] >= 0 and r["currency"] not in ("", "?")]
    if len(usable) < 4:
        return None
    currency = max({r["currency"] for r in usable}, key=lambda c: sum(r["currency"] == c for r in usable))
    same = [r for r in usable if r["currency"] == currency]
    if len(same) < 4:
        return None
    by_balance = sorted(same, key=lambda r: r["balance"], reverse=True)
    income = by_balance[0]
    rest = [r for r in same if r is not income]
    current = max(rest, key=lambda r: (r["label"] == "CURRENT", r["balance"]))
    rest.remove(current)
    savings = rest.pop(0)
    merchants = rest.pop(0)
    # Leftover accounts with money can top up the income account when it runs low.
    funding = [r["id"] for r in sorted(rest, key=lambda r: r["balance"], reverse=True) if r["balance"] > 0]
    return {"currency": currency, "current": current["id"], "savings": savings["id"],
            "income": income["id"], "merchants": merchants["id"], "funding": funding}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="save the suggestion to demo_accounts.json")
    ap.add_argument("--replace", action="store_true",
                    help="overwrite customers already in demo_accounts.json (default keeps them)")
    args = ap.parse_args()

    client, bank_id = setup()
    existing = json.loads(DEMO_ACCOUNTS_FILE.read_text()) if DEMO_ACCOUNTS_FILE.exists() else {}
    result = {"bank_id": bank_id, "customers": dict(existing.get("customers", {}))}

    for label in ("A", "B"):
        try:
            user, token = login(client, label)
        except OBPError as e:
            print(f"\n=== Customer {label}: login failed ===\n{e.body[:200]}")
            print(f"Skipping {label}. Pick another customer for {label} in .env.")
            continue
        print(f"\n=== Customer {label}: {user} ===")
        rows = survey(client, bank_id, token)
        if not rows:
            print(f"No accounts at bank {bank_id}.")
            if label == "B":
                print("Run: python -m scripts.setup_customer_b   (opens and funds accounts for B)")
            continue
        print(f"{'account_id':<28} {'balance':>12} {'cur':<4} {'tx':>4}  label")
        for r in sorted(rows, key=lambda r: r["balance"], reverse=True):
            print(f"{r['id']:<28} {r['balance']:>12.2f} {r['currency']:<4} {r['tx']:>4}  {r['label']}")

        kept = result["customers"].get(label)
        if kept and kept.get("username") == user and not args.replace:
            # Balances change after seeding, so re-suggesting would pick different accounts.
            print(f"\nKeeping the accounts already saved for {label} (use --replace to re-pick).")
            continue

        pick = suggest(rows)
        if pick is None:
            print("Not enough usable accounts in one currency to suggest a set (need 4).")
            continue
        print("\nSuggested demo accounts:")
        for role in ("current", "savings", "income", "merchants"):
            print(f"  {role:<10} {pick[role]}")
        print(f"  funding    {len(pick['funding'])} spare accounts with money")
        result["customers"][label] = {"username": user, **pick}

    if args.write and result["customers"]:
        DEMO_ACCOUNTS_FILE.write_text(json.dumps(result, indent=2))
        print(f"\nSaved {DEMO_ACCOUNTS_FILE.name}. Edit it if you prefer other accounts.")


if __name__ == "__main__":
    main()
