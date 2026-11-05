"""Visitors' pretend banks (phase 4, step 4): on SQLite, and on Postgres if TEST_DATABASE_URL is set.

    python tests/test_guest_bank.py
    $env:TEST_DATABASE_URL="postgresql://bank:bank-local-only@localhost:5432/bank"
    python tests/test_guest_bank.py                  # the same checks on Postgres too (throwaway schema)

No keys or network.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import threading
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="bank-guest-test-"))
DEMO = TMP / "demo_accounts.json"
DEMO.write_text(json.dumps({"bank_id": "gh.29.uk", "customers": {
    label: {"username": user, "currency": "GBP", "current": f"{label}-cur", "savings": f"{label}-sav",
            "income": f"{label}-inc", "merchants": f"{label}-mer", "funding": []}
    for label, user in (("A", "robert.uk.29@example.com"), ("B", "anil.uk.29@example.com"))}}))
os.environ.update({"BANK_TEST": "1", "OBP_FAKE": "1", "DEMO_ACCOUNTS_FILE": str(DEMO), "EMAIL_BACKEND": "console",
                   "OTP_EMAIL_TO": "owner@example.test", "LANGSMITH_TRACING": "false",
                   "APP_SECRET": "test-secret-1", "SERVICE_REQUESTS_FILE": str(TMP / "none.jsonl")})

import storage  # noqa: E402
from mcp_servers import common, verification  # noqa: E402
from obp import guest_bank  # noqa: E402
from obp.client import OBPError  # noqa: E402
from obp.guest_bank import BANK_ID, GUEST_BANK  # noqa: E402

OK = True


def check(label, cond):
    global OK
    OK &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label)


def refused(fn, *args, status: int | None = None) -> bool:
    try:
        fn(*args)
    except OBPError as e:
        return status is None or e.status == status
    return False


def balances(g) -> dict[str, float]:
    return {name: GUEST_BANK.balance_amount(g.token, BANK_ID, acc)[0]
            for name, acc in GUEST_BANK.accounts_of(g.guest_id).items()}


def opening() -> None:
    sam, new = GUEST_BANK.open("Sam@Example.org ", "Sam", ip="203.0.113.9")
    again, new_again = GUEST_BANK.open("sam@example.org", "Sam")
    check("a visitor gets a bank of their own; signing in again finds the same one",
          new and not new_again and again.guest_id == sam.guest_id and sam.email == "sam@example.org")
    check("the dev sign-in doesn't prove the address", sam.verified is False)
    check("... a real sign-in later marks it proven",
          GUEST_BANK.open("sam@example.org", "Sam", verified=True)[0].verified and GUEST_BANK.get(sam.guest_id).verified)
    check("expires after GUEST_TTL_HOURS (24)", 23.9 * 3600 < sam.expires - time.time() <= 24 * 3600)
    check("visible accounts: current and savings only", sorted(GUEST_BANK.accounts_of(sam.guest_id)) == ["current", "savings"])

    rows = GUEST_BANK.transactions(sam.token, BANK_ID, GUEST_BANK.accounts_of(sam.guest_id)["current"])
    days = [r["details"]["completed"][:10] for r in rows]
    today = datetime.now(timezone.utc).date()
    check("about two months of activity, newest first, none in the future",
          len(rows) > 30 and days == sorted(days, reverse=True)
          and days[0] <= today.isoformat() and days[-1] >= (today - timedelta(days=61)).isoformat())
    descs = {r["details"]["description"] for r in rows}
    check("realistic payees: salary, rent, groceries, subscriptions",
          any("Salary" in d for d in descs) and any(d.startswith("Rent") for d in descs) and "Netflix.com" in descs)
    newest = rows[0]["details"]
    check("running balance matches the account balance",
          float(newest["new_balance"]["amount"]) == balances(sam)["current"])
    profile = GUEST_BANK.my_customers(sam.token, BANK_ID)[0]
    check("profile and an address on file", profile["legal_name"] == "Sam" and profile["email"] == "sam@example.org"
          and GUEST_BANK.my_user_attributes(sam.token)[0]["name"] == "ADDRESS")


def tokens_and_isolation() -> None:
    sam = GUEST_BANK.active("sam@example.org")
    kim, _ = GUEST_BANK.open("kim@example.org", "Kim", ip="203.0.113.9")
    check("a visitor's token is signed: an edited one is refused",
          refused(GUEST_BANK.current_user, sam.token[:-2] + "00", status=401))
    check("a token for a made-up visitor is refused",
          refused(GUEST_BANK.current_user, f"guest.{'0' * 20}.{'0' * 32}", status=401))
    token = sam.token
    os.environ["APP_SECRET"] = "another-secret"
    check("tokens signed with a different APP_SECRET are refused", refused(GUEST_BANK.current_user, token, status=401))
    os.environ["APP_SECRET"] = "test-secret-1"
    sam_current = GUEST_BANK.accounts_of(sam.guest_id)["current"]
    check("one visitor can't read another's account",
          refused(GUEST_BANK.balance_amount, kim.token, BANK_ID, sam_current, status=404))
    kim_savings = GUEST_BANK.accounts_of(kim.guest_id)["savings"]
    check("... or move money into or out of it",
          refused(GUEST_BANK.transfer, kim.token, BANK_ID, sam_current, kim_savings, 5, "GBP", "x", status=404))
    check("sign-ups are counted per (hashed) address", GUEST_BANK.signups_from("203.0.113.9") == 2
          and GUEST_BANK.signups_from("198.51.100.1") == 0)
    with storage.GUESTS.connect() as db:
        stored = db.execute("SELECT ip_hash FROM guest_customers WHERE guest_id=?", (kim.guest_id,)).fetchone()[0]
    check("the address itself is never stored", "203.0.113.9" not in stored)


def transfers() -> None:
    sam = GUEST_BANK.active("sam@example.org")
    acc = GUEST_BANK.accounts_of(sam.guest_id)
    before = balances(sam)
    GUEST_BANK.transfer(sam.token, BANK_ID, acc["current"], acc["savings"], 123.45, "GBP", "To savings")
    after = balances(sam)
    check("transfer moves the money", round(before["current"] - after["current"], 2) == 123.45
          and round(after["savings"] - before["savings"], 2) == 123.45)
    top = GUEST_BANK.transactions(sam.token, BANK_ID, acc["savings"])[0]["details"]
    check("... and shows on both statements", top["description"] == "To savings" and top["value"]["amount"] == "123.45")
    check("over the limit (OBP_MAX_TRANSFER, 800): refused like the real bank",
          refused(GUEST_BANK.transfer, sam.token, BANK_ID, acc["current"], acc["savings"], 900, "GBP", "x", status=409))
    while balances(sam)["savings"] > 700:  # get savings well under the limit, so the balance is what stops it
        GUEST_BANK.transfer(sam.token, BANK_ID, acc["savings"], acc["current"],
                            min(800, round(balances(sam)["savings"] - 600, 2)), "GBP", "back")
    check("more than the balance: refused", refused(GUEST_BANK.transfer, sam.token, BANK_ID, acc["savings"],
                                                    acc["current"], balances(sam)["savings"] + 1, "GBP", "x",
                                                    status=400))

    # Ten servers move money at the same moment: every transfer counts once, and none overdraws.
    real_lock = guest_bank._LOCK
    guest_bank._LOCK = contextlib.nullcontext() if storage.is_postgres() else real_lock  # Postgres: as separate processes
    start, gate, errors = balances(sam), threading.Barrier(10), []

    def move():
        gate.wait()
        try:
            GUEST_BANK.transfer(sam.token, BANK_ID, acc["current"], acc["savings"], 10, "GBP", "race")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=move) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    guest_bank._LOCK = real_lock
    end = balances(sam)
    check("10 transfers at once: all counted, money neither made nor lost",
          not errors and round(start["current"] - end["current"], 2) == 100
          and round(end["savings"] - start["savings"], 2) == 100)


def bank_servers() -> None:
    sam = GUEST_BANK.active("sam@example.org")
    cust = common._lookup(sam.token)
    check("bank servers: a visitor's token gives a customer in the visitors' bank",
          cust.label == "G" and cust.obp is GUEST_BANK and cust.bank_id == BANK_ID
          and cust.accounts == GUEST_BANK.accounts_of(sam.guest_id))
    check("their service requests are filed separately", cust.records_key == f"guest:{sam.guest_id}")
    owner = common._lookup("fake-token-A")
    check("... and the owner's demo customer still uses the OBP bank", owner.obp is common.CLIENT and owner.label == "A")
    try:
        common._lookup(sam.token[:-1] + "x")
        check("bank servers refuse an edited visitor token", False)
    except PermissionError:
        check("bank servers refuse an edited visitor token", True)
    check("a proven address gets the codes", verification.recipient(cust) == "sam@example.org")
    kim = common._lookup(GUEST_BANK.active("kim@example.org").token)
    check("an unproven address doesn't: codes go to OTP_EMAIL_TO",
          kim.email is None and verification.recipient(kim) == "owner@example.test")
    check("codes for transfers from 500 up to the bank limit (800)",
          verification.needs_code(verification.TRANSFER, {"amount": 600}, cust.obp.max_transfer)
          and not verification.needs_code(verification.TRANSFER, {"amount": 810}, cust.obp.max_transfer)
          and not verification.needs_code(verification.TRANSFER, {"amount": 50}, cust.obp.max_transfer))


def reset_and_expiry() -> None:
    sam = GUEST_BANK.active("sam@example.org")
    acc = GUEST_BANK.accounts_of(sam.guest_id)
    GUEST_BANK.set_my_user_attribute(sam.token, "ADDRESS", "1 Changed Road, Leeds LS1 1AA")
    before = GUEST_BANK.transactions(sam.token, BANK_ID, acc["current"])
    fresh = GUEST_BANK.reset(sam.guest_id)
    after = GUEST_BANK.transactions(sam.token, BANK_ID, acc["current"])
    check("reset: same visitor and token, new activity, address back to a sample one",
          fresh.guest_id == sam.guest_id and fresh.token == sam.token and after != before
          and not any(r["details"]["description"] == "race" for r in after)
          and GUEST_BANK.my_user_attributes(sam.token)[0]["value"] != "1 Changed Road, Leeds LS1 1AA")
    kim = GUEST_BANK.active("kim@example.org")
    with storage.GUESTS.connect() as db:
        db.execute("UPDATE guest_customers SET expires=? WHERE guest_id=?", (time.time() - 1, kim.guest_id))
    check("expired: the bank stops answering", refused(GUEST_BANK.current_user, kim.token, status=401)
          and GUEST_BANK.active("kim@example.org") is None)
    check("... and is listed for deleting", GUEST_BANK.expired() == [kim.guest_id])
    GUEST_BANK.delete([kim.guest_id])
    with storage.GUESTS.connect() as db:
        left = sum(db.execute(f"SELECT COUNT(*) FROM {t} WHERE guest_id=?", (kim.guest_id,)).fetchone()[0]  # noqa: S608
                   for t in ("guest_customers", "guest_accounts", "guest_transactions", "guest_attributes"))
    check("deleting removes every row of that bank", left == 0 and GUEST_BANK.expired() == [])
    new_kim, is_new = GUEST_BANK.open("kim@example.org", "Kim")
    check("signing in after expiry gives a new bank", is_new and new_kim.guest_id != kim.guest_id)


def run_suite(name: str, env: dict) -> None:
    os.environ.update(env)
    common._CACHE.clear()
    print(f"\n--- {name}: {storage.describe()}")
    opening()
    tokens_and_isolation()
    transfers()
    bank_servers()
    reset_and_expiry()


def main() -> int:
    run_suite("SQLite", {"DATABASE_URL": "", "GUEST_DB": str(TMP / "guests.db"), "OTP_DB": str(TMP / "otp.db")})
    base = os.environ.get("TEST_DATABASE_URL", "").strip()
    if base:
        from tests.pg_scratch import scratch_url
        url, drop = scratch_url(base)
        try:
            run_suite("Postgres", {"DATABASE_URL": url})
        finally:
            storage._close_pools()
            drop()
    else:
        print("\n(Postgres skipped: set TEST_DATABASE_URL to run the same checks on Postgres)")
    print("\nALL GUEST BANK CHECKS PASSED" if OK else "\nSOME CHECKS FAILED")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
