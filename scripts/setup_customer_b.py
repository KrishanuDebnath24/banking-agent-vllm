"""Give customer B demo accounts when their sandbox login has none.

Steps:
  1. Log in customer B and open the four demo accounts (current, savings,
     income, merchants) at the bank, owned by B. Skipped if they already
     exist in demo_accounts.json.
  2. Log in customer A and send money from A's spare "funding" accounts
     into B's income account until it holds --fund. Each transfer is kept
     below OBP_MAX_TRANSFER so it completes without a second approver.
  3. Save B's accounts to demo_accounts.json.

Safe to re-run: it only tops up what is missing.

    python -m scripts.setup_customer_b            # dry run
    python -m scripts.setup_customer_b --apply
"""

from __future__ import annotations

import argparse
import json
import time
import uuid

from scripts._common import DEMO_ACCOUNTS_FILE, load_demo_accounts, login, setup
from obp.client import OBPError

NEW_ACCOUNTS = {  # role -> label shown in OBP
    "current": "Everyday Current",
    "savings": "Tax Pot Savings",
    "income": "Demo Income (hidden)",
    "merchants": "Demo Merchants (hidden)",
}
FUND_AMOUNT = 2600.00  # about one month of B's invoices plus headroom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="create accounts and move money (default is a dry run)")
    ap.add_argument("--fund", type=float, default=FUND_AMOUNT, help="target balance for B's income account")
    args = ap.parse_args()

    client, _ = setup()
    demo = load_demo_accounts()
    bank_id = demo["bank_id"]
    a = demo["customers"].get("A")
    if not a:
        print("Customer A is not in demo_accounts.json yet. Run: python -m scripts.list_accounts --write")
        return 1
    currency = a["currency"]

    user_b, token_b = login(client, "B")
    print(f"Customer B: {user_b} at {bank_id}")

    # 1. Accounts: reuse saved ones, or plan new ones.
    saved = demo["customers"].get("B", {})
    have_accounts = saved.get("username") == user_b and all(saved.get(r) for r in NEW_ACCOUNTS)
    if have_accounts:
        accounts = {r: saved[r] for r in NEW_ACCOUNTS}
        income_bal, _ = client.balance_amount(token_b, bank_id, accounts["income"])
        print(f"\nB's demo accounts already exist. Income account holds {income_bal:.2f} {currency}.")
    else:
        accounts = {r: f"demo{uuid.uuid4().hex[:20]}" for r in NEW_ACCOUNTS}
        income_bal = 0.0
        print("\nAccounts to open for B:")
        for role, acc_id in accounts.items():
            print(f"  {role:<10} {acc_id}  '{NEW_ACCOUNTS[role]}'")

    # 2. Funding: chunks from A's spare accounts, each below the approval threshold.
    needed = round(args.fund - income_bal, 2)
    transfers: list[tuple[str, float]] = []
    if needed > 0:
        _, token_a = login(client, "A")
        left = needed
        for acc_id in a.get("funding", []):
            if left <= 0:
                break
            bal, _ = client.balance_amount(token_a, bank_id, acc_id)
            take = round(min(bal, left), 2)
            for part in client.split_amount(take) if take >= 1 else []:
                transfers.append((acc_id, part))
            left = round(left - max(take, 0), 2)
        print(f"\nFunding B's income up to {args.fund:.2f} {currency} "
              f"(max {client.max_transfer:.2f} per transfer):")
        for acc_id, amt in transfers:
            print(f"  {acc_id} -> B income  {amt:.2f}")
        if left > 0:
            print(f"  WARNING: A's spare accounts are {left:.2f} short. Use a smaller --fund.")
            return 1
    else:
        print("B's income account already holds enough; no funding needed.")

    if not args.apply:
        print("\nDry run. Add --apply to make these changes.")
        return 0

    # Apply: open accounts if needed.
    if not have_accounts:
        print("\nOpening accounts...")
        for role, acc_id in accounts.items():
            try:
                client.create_account(token_b, bank_id, acc_id, NEW_ACCOUNTS[role], currency)
                print(f"  OK   {role:<10} {acc_id}")
            except OBPError as e:
                print(f"  FAIL {role:<10} {e.body[:700]}")
                print("  Stopping. Paste this error.")
                return 1
            time.sleep(0.5)
        demo["customers"]["B"] = {"username": user_b, "currency": currency, **accounts, "funding": []}
        DEMO_ACCOUNTS_FILE.write_text(json.dumps(demo, indent=2))
        print(f"Saved B's accounts to {DEMO_ACCOUNTS_FILE.name}.")

    # Apply: fund.
    if transfers:
        print("\nFunding B's income account from A...")
        for acc_id, amt in transfers:
            try:
                res = client.transfer(token_a, bank_id, acc_id, accounts["income"], amt, currency,
                                      "Internal transfer")
                print(f"  OK   {amt:.2f} [{res.get('status')}]")
            except OBPError as e:
                print(f"  FAIL {amt:.2f}: {e.body[:300]}")
                print("  Re-run this script after fixing; it only tops up what is missing.")
                return 1
            time.sleep(0.5)
        print("B is funded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
