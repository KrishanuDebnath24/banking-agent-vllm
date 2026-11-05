"""Can a customer approve their own large transfer by answering OBP's challenge?

Makes ONE transfer above the no-approval limit between customer A's HIDDEN
accounts (a spare/income account -> the "merchants" account), so the balances
the agent shows don't change. If OBP answers with a challenge, it sends the
sandbox code and reports whether the money moved, then moves it back in
small chunks.

    python -m scripts.probe_challenge                 # 900.00
    python -m scripts.probe_challenge --amount 1200
    python -m scripts.probe_challenge --code 123      # a different answer

Nothing here changes the agents. Paste the output and we'll decide on the
one-time code step from it.
"""

from __future__ import annotations

import argparse
import json
import time

from scripts._common import load_demo_accounts, login, setup
from obp.client import OBPError


def show(label: str, data) -> None:
    """Print a JSON response compactly, so the exact OBP shape is in the output."""
    text = json.dumps(data, indent=1, default=str)
    print(f"  {label}:\n    " + text.replace("\n", "\n    ")[:1500])


def challenges_of(req: dict) -> list[dict]:
    """OBP v4+ returns `challenges` (a list); older versions a single `challenge`."""
    if isinstance(req.get("challenges"), list):
        return req["challenges"]
    return [req["challenge"]] if isinstance(req.get("challenge"), dict) else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amount", type=float, default=900.0)
    ap.add_argument("--code", default="123", help="challenge answer (sandbox: 123)")
    args = ap.parse_args()

    client, _ = setup()
    demo = load_demo_accounts()
    bank_id = demo["bank_id"]
    a = demo["customers"]["A"]
    currency = a["currency"]
    username, token = login(client, "A")
    my_user_id = client.current_user(token)["user_id"]

    # Hidden accounts only: a spare account (or the income account) -> merchants.
    sources = list(a.get("funding", [])) + [a["income"]]
    src = next((acc for acc in sources if client.balance_amount(token, bank_id, acc)[0] >= args.amount), None)
    if src is None:
        print(f"No hidden account of {username} holds {args.amount:.2f} {currency}. Try a smaller --amount.")
        return 1
    dst = a["merchants"]
    before = {acc: client.balance_amount(token, bank_id, acc)[0] for acc in (src, dst)}

    print(f"Logged in as {username} (user_id {my_user_id})")
    print(f"Transfer {args.amount:.2f} {currency}: hidden account {src} -> merchants {dst}\n")

    # 1. Create the transfer (straight to the API: client.transfer refuses anything not COMPLETED).
    base = (f"/banks/{bank_id}/accounts/{src}/owner"
            f"/transaction-request-types/ACCOUNT/transaction-requests")
    body = {"to": {"bank_id": bank_id, "account_id": dst},
            "value": {"currency": currency, "amount": f"{args.amount:.2f}"},
            "description": "Challenge probe"}
    print("1. Create the transfer")
    try:
        req = client._request("POST", token, base, json=body)
    except OBPError as e:
        print(f"  OBP refused it outright: HTTP {e.status}\n  {e.body[:800]}")
        print("\nRESULT: refused before any challenge, so a code can't help at this amount."
              "\n        Paste this output.")
        return 0
    show("response", req)
    status = str(req.get("status", "")).upper()
    req_id = req.get("id")

    if status == "COMPLETED":
        print(f"\nRESULT: {args.amount:.2f} completed with no challenge. "
              f"Try a bigger --amount to reach the threshold.")
        return_money(client, token, bank_id, dst, src, args.amount, currency)
        return 0

    # 2. Answer each challenge with the sandbox code.
    challenges = challenges_of(req)
    print(f"\n2. Status {status or '?'}, {len(challenges)} challenge(s)")
    for c in challenges:
        owner = "you" if c.get("user_id") in (None, "", my_user_id) else f"ANOTHER user ({c.get('user_id')})"
        print(f"  challenge {c.get('id')}: for {owner}, type {c.get('challenge_type')}, "
              f"attempts allowed {c.get('allowed_attempts')}")
    if not challenges:
        print("  OBP gave no challenge to answer.")
    mine = [c for c in challenges if c.get("user_id") in (None, "", my_user_id)]
    if challenges and not mine:
        print("\nRESULT: the challenge belongs to another user (maker/checker). The customer"
              "\n        can't approve it with a code, so the transfer limit stays. Nothing moved.")
        return 0

    for c in mine:
        print(f"\n3. Answer challenge {c.get('id')} with {args.code!r}")
        try:
            req = client._request("POST", token, f"{base}/{req_id}/challenge",
                                  json={"id": c.get("id"), "answer": args.code})
            show("response", req)
        except OBPError as e:
            print(f"  Answer refused: HTTP {e.status}\n  {e.body[:800]}")

    # 4. Did the money move?
    time.sleep(1)
    after = {acc: client.balance_amount(token, bank_id, acc)[0] for acc in (src, dst)}
    moved = round(before[src] - after[src], 2) == round(args.amount, 2) \
        and round(after[dst] - before[dst], 2) == round(args.amount, 2)
    status = str(req.get("status", status)).upper()
    print(f"\n4. Status now {status}")
    print(f"  {src}: {before[src]:.2f} -> {after[src]:.2f}")
    print(f"  {dst}: {before[dst]:.2f} -> {after[dst]:.2f}")

    if moved:
        print("\nRESULT: YES. The customer approved their own large transfer with the code,"
              "\n        and both accounts changed. We can build the one-time code step.")
        return_money(client, token, bank_id, dst, src, args.amount, currency)
    else:
        print("\nRESULT: the money did not move. Paste this output and we'll read OBP's answer.")
    return 0


def return_money(client, token, bank_id, from_acc, to_acc, amount, currency) -> None:
    """Move the probe amount back in chunks below the no-approval limit."""
    print(f"\nMoving {amount:.2f} back in chunks of at most {client.max_transfer:.2f}:")
    for part in client.split_amount(amount):
        try:
            client.transfer(token, bank_id, from_acc, to_acc, part, currency, "Challenge probe refund")
            print(f"  {part:.2f} returned")
        except OBPError as e:
            print(f"  {part:.2f} not returned: {e.body[:200]}")
            break
        time.sleep(0.5)


if __name__ == "__main__":
    raise SystemExit(main())
