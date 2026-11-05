"""Storage checks (phase 4, step 2): every kind of record, on SQLite and on Postgres.

    python tests/test_storage.py                     # SQLite only
    $env:TEST_DATABASE_URL="postgresql://bank:bank-local-only@localhost:5432/bank"
    python tests/test_storage.py                     # SQLite, then Postgres (in a throwaway schema)

Postgres comes from  docker compose up -d db  (see SETUP.md). No keys or network otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import sys
import tempfile
import threading
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="bank-storage-test-"))
DEMO = TMP / "demo_accounts.json"
DEMO.write_text(json.dumps({"bank_id": "gh.29.uk", "customers": {
    label: {"username": user, "currency": "GBP", "current": f"{label}-cur", "savings": f"{label}-sav",
            "income": f"{label}-inc", "merchants": f"{label}-mer", "funding": []}
    for label, user in (("A", "robert.uk.29@example.com"), ("B", "anil.uk.29@example.com"))}}))
LEGACY = TMP / "service_requests.jsonl"
LEGACY.write_text("\n".join(json.dumps(r) for r in [
    {"reference": "SR-OLD001", "created_at": "2026-10-01T09:00:00+00:00", "customer": "A",
     "username": "robert.uk.29@example.com", "type": "cheque_book", "details": {"account": "current"},
     "status": "received"},
    {"reference": "SR-OLD002", "created_at": "2026-10-02T09:00:00+00:00", "customer": "A",
     "username": "robert.uk.29@example.com", "type": "address_change", "details": {"new_address": "1 Old Rd"},
     "status": "completed"}]) + "\n")
os.environ.update({"BANK_TEST": "1", "OBP_FAKE": "1", "DEMO_ACCOUNTS_FILE": str(DEMO), "EMAIL_BACKEND": "console",
                   "OTP_OUTBOX": str(TMP / "outbox.jsonl"), "OTP_EMAIL_TO": "", "LANGSMITH_TRACING": "false",
                   "SERVICE_REQUESTS_FILE": str(LEGACY)})

import storage  # noqa: E402
from mcp_servers import verification  # noqa: E402
from mcp_servers.common import Customer  # noqa: E402
from mcp_servers.services import server as services  # noqa: E402
from observability import costs  # noqa: E402

OK = True


def check(label, cond):
    global OK
    OK &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label)


def customer(label: str, username: str) -> Customer:
    return Customer(label=label, username=username, token=f"t-{label}", bank_id="gh.29.uk", currency="GBP",
                    accounts={"current": f"{label}-cur", "savings": f"{label}-sav"})


def quiet_email(to: str, subject: str, body: str) -> None:
    """Codes go to the outbox file only (the console backend would print every email)."""
    with open(TMP / "outbox.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"to": to, "subject": subject, "body": body}) + "\n")


verification.send_email = quiet_email
ROBERT = customer("A", "robert.uk.29@example.com")
ANIL = customer("B", "anil.uk.29@example.com")


def last_code() -> str:
    lines = (TMP / "outbox.jsonl").read_text().splitlines()
    return re.search(r"security code is (\d{6})", json.loads(lines[-1])["body"]).group(1)


def security_codes() -> None:
    args = {"new_address": "9 New Street, Galway H91 AB12"}
    started = verification.start(ROBERT, verification.ADDRESS, args)
    check("code sent for an address change", started["required"] and started["challenge_id"])
    wrong = verification.check(ROBERT, started["challenge_id"], "000000x")
    check("wrong code: 2 tries left", wrong == {"verified": False, "error": "wrong_code", "attempts_left": 2})
    check("another customer can't use the code",
          verification.check(ANIL, started["challenge_id"], last_code())["verified"] is False)
    check("right code verifies", verification.check(ROBERT, started["challenge_id"], last_code())["verified"])
    check("the code works for exactly this address only",
          not verification.consume(ROBERT, verification.ADDRESS, {"new_address": "elsewhere 123"}))
    check("a confirmed code can be used once", verification.consume(ROBERT, verification.ADDRESS, args))
    check("... and not twice", not verification.consume(ROBERT, verification.ADDRESS, args))

    # Ten servers try to use one confirmed code at the same moment: the database lets exactly one through.
    started = verification.start(ROBERT, verification.ADDRESS, args)
    verification.check(ROBERT, started["challenge_id"], last_code())
    real_lock, results, gate = verification._LOCK, [], threading.Barrier(10)
    verification._LOCK = contextlib.nullcontext()  # as if each thread were a different server process

    def use():
        gate.wait()
        results.append(verification.consume(ROBERT, verification.ADDRESS, args))

    threads = [threading.Thread(target=use) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    verification._LOCK = real_lock
    check("10 servers at once: the code is used exactly once", results.count(True) == 1 and len(results) == 10)

    sent = 0
    try:
        for _ in range(7):
            verification.start(ANIL, verification.ADDRESS, {"new_address": f"{sent} Quay Street, Galway H91 K2T4"})
            sent += 1
    except verification.VerificationError as e:
        check("at most 5 codes per 10 minutes per customer", sent == 5 and e.code == "too_many_codes")


def service_requests() -> None:
    services._legacy["checked"] = False  # each database gets its own one-time copy of the old file
    record = services._new_record(ROBERT, "kyc_update", {"update": "new surname"}, "received")
    services._save(record)
    mine = services.requests_of(ROBERT.username)
    check("old requests from service_requests.jsonl copied in once, newest first",
          [r["reference"] for r in mine] == [record["reference"], "SR-OLD002", "SR-OLD001"])
    check("details come back as they went in", mine[0]["details"] == {"update": "new surname"})
    services._legacy["checked"] = False
    services._save(services._new_record(ROBERT, "cheque_book", {"account": "savings"}, "received"))
    check("old file is not copied a second time", len(services.requests_of(ROBERT.username)) == 4)
    check("customers only see their own requests", services.requests_of(ANIL.username) == [])


def ledger() -> None:
    meter = costs.start_turn("A-storage1", "A")
    meter.add_llm("claude-haiku-5-5", "accounts-agent", {"input_tokens": 1000, "output_tokens": 200})
    meter.add_guard("rules", "INPUT", "enforce", ["email"], True)
    costs.end_turn(meter)
    costs.end_turn(meter)  # saving the same turn again updates it rather than adding a second turn
    q = costs.Ledger().query
    turns = q("SELECT COUNT(*), SUM(total_cost_usd), SUM(tokens) FROM turns WHERE thread_id=?", ("A-storage1",))
    check("turn saved once, with its cost and tokens",
          turns[0][0] == 1 and abs(turns[0][1] - 0.0002) < 1e-9 and turns[0][2] == 1200)
    by_day = q("SELECT substr(ts,1,10) d, COUNT(*) FROM turns WHERE ts >= ? GROUP BY d", ("2000-01-01",))
    check("cost report queries work (per day)", len(by_day) == 1 and by_day[0][1] == 1)
    guards = q("SELECT layer, findings, applied FROM guard_checks WHERE thread_id=?", ("A-storage1",))
    check("guardrail checks saved", guards[0][0] == "rules" and json.loads(guards[0][1]) == ["email"])


async def conversations() -> None:
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    class State(TypedDict):
        steps: list[str]

    def ask(state: State):
        answer = interrupt("approve?")
        return {"steps": state["steps"] + [f"approved={answer}"]}

    graph = StateGraph(State)
    graph.add_node("ask", ask)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    cfg = {"configurable": {"thread_id": "A-storage1"}}
    async with storage.checkpointer() as saver:
        await graph.compile(checkpointer=saver).ainvoke({"steps": ["start"]}, cfg)
    async with storage.checkpointer() as saver:  # "restart": a new store, same database
        app = graph.compile(checkpointer=saver)
        state = await app.aget_state(cfg)
        check("waiting confirmation survives a restart", len(state.interrupts) == 1)
        done = await app.ainvoke(Command(resume="yes"), cfg)
        check("... and can be resumed", done["steps"] == ["start", "approved=yes"])


def run_suite(name: str, env: dict) -> None:
    os.environ.update(env)
    print(f"\n--- {name}: {storage.describe()}")
    security_codes()
    service_requests()
    ledger()
    asyncio.run(conversations(), loop_factory=storage.loop_factory())
    if storage.is_postgres():
        with storage.OTP.connect() as db:
            tables = {r[0] for r in db.execute("SELECT table_name FROM information_schema.tables"
                                               " WHERE table_schema = current_schema()").fetchall()}
        check("all tables are in Postgres", {"challenges", "service_requests", "turns", "llm_calls",
                                             "guard_checks", "checkpoints"} <= tables)
        check("the password never appears in messages", "bank-local-only" not in storage.describe())


def main() -> int:
    run_suite("SQLite", {"DATABASE_URL": "", "OTP_DB": str(TMP / "otp.db"),
                         "SERVICE_REQUESTS_DB": str(TMP / "requests.db"), "TELEMETRY_DB": str(TMP / "ledger.db"),
                         "SESSION_DB": str(TMP / "sessions.db")})
    base = os.environ.get("TEST_DATABASE_URL", "").strip()
    if base:
        from tests.pg_scratch import scratch_url
        url, drop = scratch_url(base)
        try:
            (TMP / "outbox.jsonl").unlink(missing_ok=True)
            run_suite("Postgres", {"DATABASE_URL": url})
        finally:
            storage._close_pools()
            drop()
    else:
        print("\n(Postgres skipped: set TEST_DATABASE_URL to run the same checks on Postgres)")
    try:
        os.environ["DATABASE_URL"] = "mysql://nope"
        storage.database_url()
        check("a wrong DATABASE_URL is refused", False)
    except RuntimeError:
        check("a wrong DATABASE_URL is refused, not silently ignored", True)
    finally:
        os.environ["DATABASE_URL"] = ""
    print("\nALL STORAGE CHECKS PASSED" if OK else "\nSOME CHECKS FAILED")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
