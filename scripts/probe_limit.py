"""Find the largest transfer that completes without a second approver.

Moves small test amounts between customer A's HIDDEN accounts only
(a spare "funding" account -> A's "merchants" account), increasing the
amount until OBP asks for approval. Then checks whether a £1 transfer
from A to B's income account completes.

    python -m scripts.probe_limit

Set the result as OBP_MAX_TRANSFER in .env.
"""

from __future__ import annotations

import time

from scripts._common import load_demo_accounts, login, setup
from obp.client import OBPError

STEPS = [1, 100, 300, 500, 600, 700, 750, 800, 850, 900]


def pick_source(client, token, bank_id, funding: list[str], amount: float) -> str | None:
    for acc_id in funding:
        bal, _ = client.balance_amount(token, bank_id, acc_id)
        if bal >= amount:
            return acc_id
    return None


def main() -> int:
    client, _ = setup()
    client.max_transfer = 10_000  # let the probe try amounts above the current cap
    demo = load_demo_accounts()
    bank_id = demo["bank_id"]
    a = demo["customers"]["A"]
    currency = a["currency"]
    _, token = login(client, "A")

    print(f"Probing own-account transfers ({currency}), hidden accounts only:")
    best = 0.0
    for amount in STEPS:
        src = pick_source(client, token, bank_id, a.get("funding", []), amount)
        if src is None:
            print(f"  {amount:>6}: no spare account holds this much; stopping")
            break
        try:
            res = client.transfer(token, bank_id, src, a["merchants"], amount, currency, "Limit probe")
            print(f"  {amount:>6}: COMPLETED")
            best = amount
        except OBPError as e:
            print(f"  {amount:>6}: needs approval -> {e.body[:120]}")
            break
        time.sleep(0.5)

    b = demo["customers"].get("B")
    cross_ok = None
    if b:
        print("\nProbing a 1.00 transfer from customer A to customer B's income account:")
        src = pick_source(client, token, bank_id, a.get("funding", []), 1)
        try:
            client.transfer(token, bank_id, src, b["income"], 1, currency, "Limit probe")
            print("  COMPLETED")
            cross_ok = True
        except OBPError as e:
            print(f"  needs approval -> {e.body[:120]}")
            cross_ok = False

    print("\nResult:")
    if best:
        print(f"  Largest transfer that completed: {best:.2f} {currency}")
        print(f"  Set in .env:  OBP_MAX_TRANSFER={best:g}")
    else:
        print("  Even 1.00 needed approval for own-account transfers. Paste this output.")
    if cross_ok is False:
        print("  Transfers from A to B need approval at any size, so B can't be funded from A.")
        print("  Paste this output and we'll pick another way to give B money.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
