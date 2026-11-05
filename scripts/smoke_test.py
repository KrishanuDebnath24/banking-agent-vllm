"""Smoke test for the OBP sandbox.

Checks, in order:
  1. Customer A and customer B can log in (Direct Login).
  2. Who each token belongs to.
  3. Customer A's accounts, balances and recent transactions.
  4. Customer B is REFUSED when asking for customer A's transactions
     (this is the authorisation demo for the agent).

Run from the repo root:   python -m scripts.smoke_test
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obp.client import OBPClient, OBPError  # noqa: E402


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    load_dotenv()
    client = OBPClient.from_env()
    bank_id = os.environ.get("OBP_BANK_ID", "gh.29.uk")

    # 1. Log in both customers
    section("1. Direct Login")
    tokens = {}
    for label in ("A", "B"):
        user = os.environ[f"CUSTOMER_{label}_USERNAME"]
        try:
            tokens[label] = client.login(user, os.environ[f"CUSTOMER_{label}_PASSWORD"])
            print(f"Customer {label} ({user}): logged in")
        except OBPError as e:
            print(f"Customer {label} ({user}): LOGIN FAILED -> {e}")
            print("  Check OBP_BASE_URL, the consumer key, and that this sandbox login still exists.")
            return 1

    # 2. Who is each token?
    section("2. Current user")
    for label, token in tokens.items():
        me = client.current_user(token)
        print(f"Customer {label}: user_id={me.get('user_id')} username={me.get('username')}")

    # 3. Customer A's accounts, balances, transactions
    section(f"3. Customer A at bank {bank_id}")
    try:
        accounts = client.accounts_held(tokens["A"], bank_id)
    except OBPError as e:
        print(f"Could not list accounts: {e}")
        print("  Try another OBP_BANK_ID; the API Explorer shows which bank this user belongs to.")
        return 1

    if not accounts:
        print("No accounts found. Set OBP_BANK_ID to the bank this customer belongs to.")
        return 1

    for acc in accounts:
        print(f"- account {acc.get('id')} label={acc.get('label')}")

    first_account = accounts[0]["id"]
    try:
        print("Balances:", client.balances(tokens["A"], bank_id, first_account))
    except OBPError as e:
        print(f"Balances call failed (may not exist in {client.version}): {e}")

    txns = client.transactions(tokens["A"], bank_id, first_account)
    print(f"{len(txns)} transactions; first 3:")
    for t in txns[:3]:
        details = t.get("details", {})
        value = details.get("value", {})
        print(f"  {details.get('completed')}  {value.get('amount')} {value.get('currency')}  {details.get('description')}")

    # 4. Authorisation check: B must NOT see A's account
    section("4. Cross-customer access (B -> A's account)")
    try:
        client.transactions(tokens["B"], bank_id, first_account)
        print("WARNING: customer B could read customer A's transactions. Pick a different pair of customers.")
        return 1
    except OBPError as e:
        print(f"Refused as expected: HTTP {e.status}")
        print(f"  {e.body[:200]}")

    print("\nAll checks passed. The OBP client is ready for agent tools.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
