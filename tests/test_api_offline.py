"""Offline end-to-end test of the web API (phase 4).

Real API + real bank servers + the fixed eval bank and visitors' pretend banks, with a scripted
model. No keys or network.

    python tests/test_api_offline.py                       # records in SQLite files
    $env:TEST_DATABASE_URL="postgresql://bank:bank-local-only@localhost:5432/bank"
    python tests/test_api_offline.py                       # the same checks on Postgres (own scratch schema)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import tempfile
import time
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="bank-api-test-"))
OFFSET = 800
os.environ.update({"BANK_TEST": "1", "LANGSMITH_TRACING": "false", "EVAL_PORT_OFFSET": str(OFFSET),
                   "OTP_OUTBOX": str(TMP / "outbox.jsonl"), "OBP_FAKE": "1", "AUTH_MODE": "dev",
                   "OTP_EMAIL_TO": "owner@example.test", "APP_SECRET": "api-test-secret",
                   "SESSION_DB": str(TMP / "sessions.db"), "GUEST_DB": str(TMP / "guests.db"),
                   "WEB_DB": str(TMP / "web.db"), "VERIFY_URL": f"http://127.0.0.1:{8104 + OFFSET}",
                   "CLIENT_IP_HEADER": "CloudFront-Viewer-Address",
                   "MAINTENANCE_SECONDS": "3600"})   # housekeeping is run by hand below

PG_BASE = os.environ.get("TEST_DATABASE_URL", "").strip()
DROP_PG = None
if PG_BASE:  # API, conversation store and bank servers all on Postgres, in a throwaway schema
    from tests.pg_scratch import scratch_url  # noqa: E402
    PG_URL, DROP_PG = scratch_url(PG_BASE)
    os.environ.update({"DATABASE_URL": PG_URL, "EVAL_DATABASE_URL": PG_URL})
else:
    os.environ.update({"DATABASE_URL": "", "EVAL_DATABASE_URL": ""})

from evals import harness  # noqa: E402

work = harness.configure_env()
os.environ.update({"DEMO_ACCOUNTS_FILE": os.environ["EVAL_DEMO"], "TELEMETRY_DB": str(TMP / "telemetry.db"),
                   "SERVICE_REQUESTS_DB": str(work / "requests.db"),   # the same files the bank servers use
                   "OTP_DB": str(work / "otp.db")})

import httpx  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import api.app as api_module  # noqa: E402
import storage  # noqa: E402
from api import state  # noqa: E402
from guardrails.middleware import BLOCKED_SECRET_REPLY  # noqa: E402
from obp.guest_bank import BANK_ID, GUEST_BANK  # noqa: E402

OK = True


def check(label, cond):
    global OK
    OK &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label)


class Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


class Stuck(Fake):
    """A model that never answers, to see what a restart does to an answer in progress."""
    async def _agenerate(self, *args, **kwargs):
        await asyncio.sleep(3600)


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "id": f"id-{uuid.uuid4().hex}", "args": args}])


def said(text):
    return AIMessage(content=text)


# Every agent (one per person, per app copy) takes its next reply from this one script, in test order.
def _play(items):
    """The script, with some replies worked out when they're used (a dataset id the bank just made)."""
    for item in items:
        yield item() if callable(item) else item


def latest_dataset() -> str:
    with storage.CHARTS.connect() as db:
        return db.execute("SELECT dataset_id FROM chart_datasets ORDER BY created DESC LIMIT 1").fetchone()[0]


SCRIPT = _play([
    # Robert (owner, A): balances
    call("task", subagent_type="accounts-agent", description="List the customer's balances."),
    call("list_my_accounts"), said("current £5,234.77, savings £1,201.12"),
    said("Your **current** account has £5,234.77 and your savings £1,201.12."),
    # Robert: £600 transfer, approved with a code
    call("task", subagent_type="accounts-agent", description="Transfer £600.00 from current to savings."),
    call("transfer_between_my_accounts", from_account="current", to_account="savings", amount=600),
    said("Transferred £600.00."), said("Done: £600.00 moved from current to savings."),
    # Robert: £20 transfer, declined
    call("task", subagent_type="accounts-agent", description="Transfer £20.00 from current to savings."),
    call("transfer_between_my_accounts", from_account="current", to_account="savings", amount=20),
    said("The customer declined."), said("You chose not to go ahead, so I haven't made the transfer."),
    # Robert: cheque book, approved on a second copy of the app
    call("task", subagent_type="service-agent", description="Order a cheque book for the current account."),
    call("request_cheque_book", account="current"),
    said("Cheque book ordered."), said("Your cheque book is on its way."),
    # Sam (visitor): balances
    call("task", subagent_type="accounts-agent", description="List the customer's balances."),
    call("list_my_accounts"), said("listed"), said("Here are your balances."),
    # Sam: £600 transfer with a code
    call("task", subagent_type="accounts-agent", description="Transfer £600.00 from current to savings."),
    call("transfer_between_my_accounts", from_account="current", to_account="savings", amount=600),
    said("Transferred £600.00."), said("Done: £600.00 moved to savings."),
    # Sam: cheque book
    call("task", subagent_type="service-agent", description="Order a cheque book for the current account."),
    call("request_cheque_book", account="current"), said("Ordered."), said("Your cheque book is on its way."),
    # Vera (visitor whose sign-in proved her address): address change, code to her own inbox
    call("task", subagent_type="service-agent", description="Change address to 1 New Street, Leeds LS1 1AA."),
    call("change_address", new_address="1 New Street, Leeds LS1 1AA"),
    said("Changed."), said("Your address is now 1 New Street, Leeds LS1 1AA."),
    # Chloe (visitor): a chart takes two specialists: the figures, then the drawing
    call("task", subagent_type="transactions-agent", description="Grocery spending for each month, for a chart."),
    call("spending_by_month", category="groceries", label="Groceries"),
    lambda: said(f"Monthly grocery totals saved as dataset {latest_dataset()}."),
    lambda: call("task", subagent_type="chart-agent", description=f"Draw dataset {latest_dataset()} as columns."),
    lambda: call("column_chart", dataset_id=latest_dataset()),
    said("Drew a column chart."), said("Here's your grocery spending by month."),
    # Chloe: the same figures as a line: only the chart specialist, the bank isn't asked again
    lambda: call("task", subagent_type="chart-agent", description=f"Draw dataset {latest_dataset()} as a line."),
    lambda: call("line_chart", dataset_id=latest_dataset()),
    said("Drew a line chart."), said("Here it is as a line."),
    # Chloe: where her money goes, as a donut
    call("task", subagent_type="transactions-agent", description="Spending by category, for a donut chart."),
    call("spending_breakdown", by="category"),
    lambda: said(f"Spending by category saved as dataset {latest_dataset()}."),
    lambda: call("task", subagent_type="chart-agent", description=f"Draw dataset {latest_dataset()} as a donut."),
    lambda: call("donut_chart", dataset_id=latest_dataset()),
    said("Drew a donut chart."), said("Here's where your money goes."),
])


def model():
    return Fake(messages=SCRIPT)


def outbox() -> list[dict]:
    path = TMP / "outbox.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def latest_code() -> str:
    return re.search(r"security code is (\d{6})", outbox()[-1]["body"]).group(1)


async def tool(token, server, name, args=None):
    from agents.coordinator import load_mcp_tools
    tools = {t.name: t for t in (await load_mcp_tools(token))[server]}
    out = await tools[name].ainvoke(args or {})
    return json.loads(out[0]["text"] if isinstance(out, list) else out)


async def balance(token, account):
    return (await tool(token, "accounts", "get_balance", {"account": account}))["balance"]


async def requests_of(token):
    return (await tool(token, "service", "list_my_service_requests"))["requests"]


async def wait(client, turn_id, seen_running: list | None = None) -> dict:
    """Ask for progress until the answer is ready, like the page does: long checks that the server
    holds until there's news."""
    seen = 0
    for _ in range(100):
        r = (await client.get(f"/api/turns/{turn_id}", params={"wait": 5, "seen": seen})).json()
        if r["status"] != "running":
            return r
        seen = len(r["steps"])
        if seen_running is not None:
            seen_running.append(r)
    raise TimeoutError(turn_id)


async def ask(client, message, thread=None) -> dict:
    r = await client.post("/api/chat", json={"message": message, "thread_id": thread})
    assert r.status_code == 202, r.text
    return await wait(client, r.json()["turn_id"])


async def confirm(client, thread, approve=True, code=None) -> dict:
    r = await client.post("/api/confirm", json={"thread_id": thread, "approve": approve, "code": code})
    if r.status_code == 202:
        return await wait(client, r.json()["turn_id"])
    return {"http": r.status_code, **r.json()}


def client(app, cookies=None, ip=None):
    headers = {"CloudFront-Viewer-Address": f"{ip}:443"} if ip else {}
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bank.test",
                             cookies=cookies, headers=headers)


async def owner_flow(app) -> str:
    async with client(app) as a, client(app) as b:
        cfg = (await a.get("/api/config")).json()
        check("sign-in options list the demo customers, and visitors' limits",
              [c["name"] for c in cfg["customers"]] == ["Robert", "Anil"] and cfg["visitors"]["bank_hours"] == 24)
        check("not signed in: /api/me is 401", (await a.get("/api/me")).status_code == 401)
        check("not signed in: chat is refused", (await a.post("/api/chat", json={"message": "hi"})).status_code == 401)

        r = await a.post("/api/login", json={"customer": "A"})
        cookie = r.headers.get("set-cookie", "")
        check("sign in as Robert; session cookie is HttpOnly and SameSite=Strict",
              r.status_code == 200 and "httponly" in cookie.lower() and "samesite=strict" in cookie.lower())
        check("the browser never gets the bank token", "fake-token" not in r.text and "fake-token" not in cookie)
        with storage.WEB.connect() as db:
            stored = db.execute("SELECT sid_hash FROM web_sessions").fetchall()
        sid = a.cookies.get("bank_session")
        check("the database keeps only a hash of the session id", stored and all(sid not in row[0] for row in stored))

        r = await a.post("/api/chat", json={"message": "What are my balances?"})
        check("chat replies at once with a turn to check back on (202)",
              r.status_code == 202 and r.json()["status"] == "running" and re.match(r"^[0-9a-f]{16}$", r.json()["turn_id"]))
        thread = r.json()["thread_id"]
        busy = await a.post("/api/chat", json={"message": "and again?", "thread_id": thread})
        check("a second message while one is being answered: 409 with the running turn",
              busy.status_code == 409 and busy.json()["turn_id"] == r.json()["turn_id"])
        started = time.monotonic()
        r = (await a.get(f"/api/turns/{r.json()['turn_id']}", params={"wait": 20, "seen": 99})).json()
        check("a long check returns as soon as the answer is ready, not after 20 seconds",
              r["status"] == "answered" and time.monotonic() - started < 10)
        started = time.monotonic()
        again = (await a.get(f"/api/turns/{r['turn_id']}", params={"wait": 20})).json()
        check("... and at once for an answer that's already there", again["status"] == "answered"
              and time.monotonic() - started < 2)
        check("the answer arrives with the balances", r["status"] == "answered" and "5,234.77" in r["answer"])
        check("what happened: specialist and tool listed",
              r["steps"][0]["label"] == "Accounts specialist" and r["steps"][1]["name"] == "list_my_accounts")
        check("cost reported for the turn", r["cost"].startswith("$"))

        before = await balance("fake-token-A", "savings")
        r = await ask(a, "Move £600 to savings", thread)
        check("transfer stops at a confirmation slip", r["status"] == "needs_confirmation"
              and r["slip"]["title"] == "Transfer between your accounts" and ["Amount", "£600.00"] in r["slip"]["rows"])
        busy = await a.post("/api/chat", json={"message": "hello?", "thread_id": thread})
        check("can't chat past a waiting slip (409, slip returned)", busy.status_code == 409 and busy.json()["slip"])

        r = await confirm(a, thread)
        check("approve asks for the emailed code (owner's codes go to OTP_EMAIL_TO)",
              r["status"] == "code_required" and r["sent_to"] == "o***@example.test")
        hist = (await a.get(f"/api/history?thread_id={thread}")).json()
        check("after a reload, the page gets the slip and the code box back",
              hist["slip"]["title"] == "Transfer between your accounts" and hist["code"]["sent_to"] == "o***@example.test")
        r = await confirm(a, thread, code="000000x")
        check("wrong code: tries left shown", r["status"] == "wrong_code" and r["attempts_left"] == 2)
        code = latest_code()
        r = await confirm(a, thread, code=code)
        after = await balance("fake-token-A", "savings")
        check("right code: transfer done and answered", r["status"] == "answered" and round(after - before, 2) == 600.0)

        await ask(a, "Move £20 to savings", thread)
        r = await confirm(a, thread, approve=False)
        check("decline: nothing moved, answer says so", r["status"] == "answered"
              and "chose not to go ahead" in r["answer"] and await balance("fake-token-A", "savings") == after)

        r = await ask(a, "my card is 4111 1111 1111 1111, why declined?", thread)
        check("card number blocked by the guardrail", r["answer"] == BLOCKED_SECRET_REPLY
              and any("card_number" in g for g in r["guard"]))

        hist = (await a.get(f"/api/history?thread_id={thread}")).json()
        texts = " ".join(m["text"] for m in hist["messages"])
        check("history has the conversation", "What are my balances?" in texts and "5,234.77" in texts)
        check("history never contains the card number or the code", "4111" not in texts and code not in texts)

        await b.post("/api/login", json={"customer": "B"})
        check("Anil can't read Robert's conversation",
              (await b.get(f"/api/history?thread_id={thread}")).status_code == 404)
        check("Anil can't confirm on Robert's conversation",
              (await b.post("/api/confirm", json={"thread_id": thread, "approve": True})).status_code == 404)
        check("Anil can't see Robert's answers", (await b.get(f"/api/turns/{r['turn_id']}")).status_code == 404)
        check("made-up thread ids are refused", (await a.get("/api/history?thread_id=A-../../x")).status_code == 404)

        r = await ask(a, "Order a cheque book for my current account", thread)
        check("cheque book waits on its slip", r["status"] == "needs_confirmation"
              and r["slip"]["title"] == "Order a cheque book")
        mine = (await a.get("/api/conversations")).json()["conversations"]
        check("your conversations: titled with the first message, and marked as waiting for an approval",
              mine and mine[0]["thread_id"] == thread and mine[0]["title"] == "What are my balances?"
              and mine[0]["waiting"] is True and mine[0]["updated"] > time.time() - 60)
        check("... Anil's list doesn't show Robert's conversations",
              all(c["thread_id"] != thread for c in (await b.get("/api/conversations")).json()["conversations"]))
        before_requests = len(await requests_of("fake-token-A"))

        # A second copy of the app (a restart, or autoscaling): same database, nothing in memory.
        app2 = api_module.create_app(model_factory=model)
        async with app2.router.lifespan_context(app2), client(app2, cookies=a.cookies) as a2:
            check("another copy: still signed in (sessions are in the database)",
                  (await a2.get("/api/me")).json().get("customer") == "A")
            hist = (await a2.get(f"/api/history?thread_id={thread}")).json()
            check("another copy: the conversation and the waiting slip are there",
                  "5,234.77" in " ".join(m["text"] for m in hist["messages"])
                  and (hist["slip"] or {}).get("title") == "Order a cheque book")
            check("another copy: still can't chat past the slip",
                  (await a2.post("/api/chat", json={"message": "hello?", "thread_id": thread})).status_code == 409)
            started = await a2.post("/api/confirm", json={"thread_id": thread, "approve": True})
            twice = await a.post("/api/confirm", json={"thread_id": thread, "approve": True})
            check("approving the same slip on two copies at once: only one goes ahead",
                  started.status_code == 202 and twice.status_code == 409)
            r = await wait(a, started.json()["turn_id"])   # asked on the first copy: any copy can answer
            reqs = await requests_of("fake-token-A")
            check("... and the cheque book is ordered once", r["status"] == "answered"
                  and len(reqs) == before_requests + 1 and reqs[0]["type"] == "cheque_book")

        page = await a.get("/")
        check("chat page served", page.status_code == 200 and "Demo Bank" in page.text)
        os.environ["OWNER_EMAILS"] = "owner.krish@example.org"
        privacy = await a.get("/privacy.html")
        check("privacy notice served, with a contact address (needed to publish Google sign-in)",
              privacy.status_code == 200 and "Privacy notice" in privacy.text
              and "mailto:owner.krish@example.org" in privacy.text and "__CONTACT__" not in privacy.text)
        os.environ["CONTACT_EMAIL"] = "privacy@example.org"
        check("... CONTACT_EMAIL can name a different address",
              "mailto:privacy@example.org" in (await a.get("/privacy")).text)
        os.environ.update({"OWNER_EMAILS": "", "CONTACT_EMAIL": ""})
        check("security headers set", "frame-ancestors 'none'" in page.headers.get("content-security-policy", "")
              and page.headers.get("x-content-type-options") == "nosniff")

        api_module.RATE_LIMIT = 3
        codes = [(await b.get(f"/api/history?thread_id=B-{'0' * 8}")).status_code for _ in range(5)]
        check("rate limit kicks in", 429 in codes)
        api_module.RATE_LIMIT = 30
        return thread


async def visitor_flow(app, owner_thread: str) -> None:
    async with client(app, ip="203.0.113.9") as sam, client(app, ip="203.0.113.9") as robert:
        await robert.post("/api/login", json={"customer": "A"})
        check("a visitor must give a real-looking email",
              (await sam.post("/api/login", json={"email": "not-an-email"})).status_code == 400)
        r = await sam.post("/api/login", json={"email": "Sam.Taylor@Example.org"})
        check("a visitor signs in with an email and gets their own bank",
              r.status_code == 200 and r.json()["kind"] == "guest" and r.json()["name"] == "Sam")
        me = (await sam.get("/api/me")).json()
        check("/api/me: visitor, 30 messages left, bank kept 24 hours",
              me["kind"] == "guest" and me["messages_left"] == 30 and 23.9 * 3600 < me["expires"] - time.time() <= 86400)
        guest = GUEST_BANK.active("sam.taylor@example.org")
        token = guest.token
        check("the visitor's bank starts with its own balances",
              await balance(token, "current") == GUEST_BANK.balance_amount(token, BANK_ID,
                                                                           GUEST_BANK.accounts_of(guest.guest_id)["current"])[0])

        progress: list = []
        r = await sam.post("/api/chat", json={"message": "What are my balances?"})
        thread = r.json()["thread_id"]
        r = await wait(sam, r.json()["turn_id"], progress)
        check("visitor's conversation ids are their own", re.match(r"^g[0-9a-f]{10}-[0-9a-f]{8}$", thread)
              and thread.startswith(me["customer"]))
        check("visitor gets an answer", r["status"] == "answered" and r["steps"][1]["name"] == "list_my_accounts")

        before = await balance(token, "savings")
        owner_before = await balance("fake-token-A", "savings")
        r = await ask(sam, "Move £600 to savings", thread)
        r = await confirm(sam, thread)
        check("visitor's code: their address isn't proven (dev sign-in), so it goes to OTP_EMAIL_TO",
              r["status"] == "code_required" and r["sent_to"] == "o***@example.test"
              and outbox()[-1]["to"] == "owner@example.test")
        r = await confirm(sam, thread, code=latest_code())
        check("visitor's transfer happens in their own bank only", r["status"] == "answered"
              and round(await balance(token, "savings") - before, 2) == 600
              and await balance("fake-token-A", "savings") == owner_before)

        await ask(sam, "Order a cheque book", thread)
        r = await confirm(sam, thread)
        mine = [x["reference"] for x in await requests_of(token)]
        owners = [x["reference"] for x in await requests_of("fake-token-A")]
        check("visitor's cheque book is filed as theirs, not the owner's",
              r["status"] == "answered" and len(mine) == 1 and mine[0] not in owners)

        check("the owner can't open the visitor's conversation",
              (await robert.get(f"/api/history?thread_id={thread}")).status_code == 404)
        check("the visitor can't open the owner's conversation",
              (await sam.get(f"/api/history?thread_id={owner_thread}")).status_code == 404)
        a_current = json.loads(Path(os.environ["EVAL_DEMO"]).read_text())["customers"]["A"]["current"]
        denied = await tool(token, "accounts", "get_balance", {"account": a_current})
        check("bank servers: a visitor can't reach the owner's account, even by its id", denied["error"] == "access_denied")
        forged = await tool(token[:-2] + "00", "accounts", "list_my_accounts")
        check("bank servers: an edited visitor token is refused", forged["error"] == "not_authorised")

        # Limits
        os.environ["GUEST_DAILY_MESSAGES"] = "3"   # Sam has sent 3
        r = await sam.post("/api/chat", json={"message": "Hi", "thread_id": thread})
        check("visitor's daily message limit", r.status_code == 429 and r.json()["limit"] == "messages")
        os.environ["GUEST_DAILY_MESSAGES"] = "30"
        os.environ["DAILY_SPEND_CAP_USD"] = "0"
        r = await sam.post("/api/chat", json={"message": "Hi", "thread_id": thread})
        check("visitors' daily spending cap", r.status_code == 429 and r.json()["limit"] == "budget")
        r = await robert.post("/api/chat", json={"message": "my card is 4111 1111 1111 1111"})
        check("... the owner isn't limited", r.status_code == 202)
        await wait(robert, r.json()["turn_id"])
        listed = (await robert.get("/api/conversations")).json()["conversations"]
        check("a new conversation goes to the top of the list, its title without the card number",
              listed[0]["thread_id"] == r.json()["thread_id"] and "4111" not in listed[0]["title"]
              and "card" in listed[0]["title"] and listed[1]["thread_id"] == owner_thread)
        with storage.WEB.connect() as db:   # as if from before the list existed
            db.execute("DELETE FROM chat_threads WHERE thread_id=?", (owner_thread,))
        listed = (await robert.get("/api/conversations")).json()["conversations"]
        check("... a conversation from before the list existed gets its title from its first message",
              any(c["thread_id"] == owner_thread and c["title"] == "What are my balances?" for c in listed))
        os.environ["DAILY_SPEND_CAP_USD"] = "2.00"

        os.environ["GUEST_SIGNUPS_PER_IP_HOUR"] = "1"
        async with client(app, ip="203.0.113.9") as other, client(app, ip="198.51.100.7") as elsewhere:
            r = await other.post("/api/login", json={"email": "someone.new@example.org"})
            check("new demo banks per address per hour are limited", r.status_code == 429 and r.json()["limit"] == "signups")
            check("... an existing visitor can still sign in from there",
                  (await other.post("/api/login", json={"email": "sam.taylor@example.org"})).status_code == 200)
            check("... and other addresses aren't affected (address read from CLIENT_IP_HEADER)",
                  (await elsewhere.post("/api/login", json={"email": "someone.new@example.org"})).status_code == 200)
        os.environ["GUEST_SIGNUPS_PER_IP_HOUR"] = "5"

        from starlette.requests import Request
        os.environ["CLIENT_IP_HEADER"] = "X-Forwarded-For"
        behind_alb = Request({"type": "http", "client": ("10.20.0.9", 4321),
                              "headers": [(b"x-forwarded-for", b"6.6.6.6, 203.0.113.50")]})
        check("behind a load balancer: the address it added counts, not one the visitor made up",
              api_module.client_ip(behind_alb) == "203.0.113.50")
        os.environ["CLIENT_IP_HEADER"] = "CloudFront-Viewer-Address"

        # Start over
        old_rows = GUEST_BANK.transactions(token, BANK_ID, GUEST_BANK.accounts_of(guest.guest_id)["current"])
        r = await sam.post("/api/guest/reset")
        hist = (await sam.get(f"/api/history?thread_id={thread}")).json()
        check("reset: new activity, conversations and requests cleared",
              r.status_code == 200 and hist["messages"] == [] and await requests_of(token) == []
              and (await sam.get("/api/conversations")).json()["conversations"] == []
              and GUEST_BANK.transactions(token, BANK_ID, GUEST_BANK.accounts_of(guest.guest_id)["current"]) != old_rows)
        check("the owner has no reset", (await robert.post("/api/guest/reset")).status_code == 404)


async def verified_visitor_and_expiry(app) -> None:
    vera, _ = GUEST_BANK.open("vera@example.org", "Vera", verified=True)  # as if she signed in with Google
    async with client(app, ip="198.51.100.20") as v:
        await v.post("/api/login", json={"email": "vera@example.org"})
        r = await ask(v, "Change my address to 1 New Street, Leeds LS1 1AA")
        thread = r["thread_id"]
        r = await confirm(v, thread)
        check("a visitor whose sign-in proved her address gets codes in her own inbox",
              r["status"] == "code_required" and r["sent_to"] == "v***@example.org"
              and outbox()[-1]["to"] == "vera@example.org")
        r = await confirm(v, thread, code=latest_code())
        profile = await tool(vera.token, "service", "get_my_profile")
        check("... and her address changes in her bank", r["status"] == "answered"
              and profile["address"] == "1 New Street, Leeds LS1 1AA" and len(await requests_of(vera.token)) == 1)

        from mcp_servers import datasets
        vera_figures = datasets.save(f"guest:{vera.guest_id}", "monthly", {"label": "test"})
        with storage.GUESTS.connect() as db:   # 24 hours later
            db.execute("UPDATE guest_customers SET expires=? WHERE guest_id=?", (time.time() - 1, vera.guest_id))
        cp = None
        async with storage.checkpointer() as saver:
            cp = await saver.aget_tuple({"configurable": {"thread_id": thread, "checkpoint_ns": ""}})
        done = await app.state.maintenance_once()
        check("housekeeping deletes expired visitors", done["expired_visitors"] == 1)
        check("... signed out", (await v.get("/api/me")).status_code == 401)
        with storage.WEB.connect() as db:
            turns = db.execute("SELECT COUNT(*) FROM chat_turns WHERE owner_key=?", (vera.owner_key,)).fetchone()[0]
        async with storage.checkpointer() as saver:
            gone = await saver.aget_tuple({"configurable": {"thread_id": thread, "checkpoint_ns": ""}})
        with storage.REQUESTS.connect() as db:
            reqs = db.execute("SELECT COUNT(*) FROM service_requests WHERE username=?",
                              (f"guest:{vera.guest_id}",)).fetchone()[0]
        check("... with their conversations, requests, figures saved for charts, and bank",
              cp is not None and gone is None and turns == 0 and reqs == 0 and GUEST_BANK.get(vera.guest_id) is None
              and datasets.load(f"guest:{vera.guest_id}", vera_figures) is None)
        old = datasets.save("robert.uk.29@example.com", "monthly", {"label": "test"})
        with storage.CHARTS.connect() as db:
            db.execute("UPDATE chart_datasets SET created=? WHERE dataset_id=?", (time.time() - 8 * 86400, old))
        done = await app.state.maintenance_once()
        check("housekeeping deletes figures saved for charts after 7 days", done["old_chart_data"] >= 1
              and datasets.load("robert.uk.29@example.com", old) is None)

    with storage.OTP.connect() as db:
        codes_before = db.execute("SELECT COUNT(*) FROM challenges").fetchone()[0]
        db.execute("UPDATE challenges SET created=?", (time.time() - 2 * 86400,))
    done = await app.state.maintenance_once()
    with storage.OTP.connect() as db:
        codes_after = db.execute("SELECT COUNT(*) FROM challenges").fetchone()[0]
    check("housekeeping deletes security code records after a day (they name the customer's email)",
          codes_before > 0 and codes_after == 0 and done["old_security_codes"] == codes_before)

    # An answer whose app copy stopped without saying so: housekeeping marks it, so nobody waits forever.
    turn_id, _ = state.start_chat("A-0badc0de", "A")
    with storage.WEB.connect() as db:
        db.execute("UPDATE chat_turns SET updated=? WHERE turn_id=?", (time.time() - 3600, turn_id))
    await app.state.maintenance_once()
    found = state.get_turn(turn_id)
    check("housekeeping: an answer left running by a stopped copy is marked failed, with a message",
          found["status"] == "failed" and "restarted" in found["result"]["error"])


async def chart_flow(app) -> None:
    """Charts: the transactions specialist saves the figures (a dataset id), the chart specialist
    draws that id. A different form of the same chart needs only the chart specialist."""
    from datetime import date, timedelta

    from api import charts
    async with client(app, ip="198.51.100.30") as c:
        await c.post("/api/login", json={"email": "chloe@example.org"})
        guest = GUEST_BANK.active("chloe@example.org")
        token = guest.token

        r1 = await ask(c, "Can you generate me a chart of grocery spending each month for the whole year?")
        thread = r1["thread_id"]
        chart1 = (r1.get("charts") or [None])[0]
        tools1 = [s["name"] for s in r1["steps"] if s["kind"] == "tool"]
        who1 = [s["name"] for s in r1["steps"] if s["kind"] == "specialist"]
        ds = next(s["args"]["dataset_id"] for s in r1["steps"] if s["name"] == "column_chart")
        check("a chart: the transactions specialist saves the figures, then the chart specialist draws them",
              r1["status"] == "answered" and who1 == ["transactions-agent", "chart-agent"]
              and tools1 == ["spending_by_month", "column_chart"] and chart1 and chart1["form"] == "column")
        r2 = await ask(c, "Show that as a line chart", thread)
        chart2 = (r2.get("charts") or [None])[0]
        check("... another form of the same figures: only the chart specialist, the bank isn't asked again",
              r2["status"] == "answered" and [s["name"] for s in r2["steps"] if s["kind"] == "specialist"] == ["chart-agent"]
              and [(s["name"], s["args"]) for s in r2["steps"] if s["kind"] == "tool"] == [("line_chart", {"dataset_id": ds})]
              and chart2 and chart2["form"] == "line"
              and (chart2["labels"], chart2["series"]) == (chart1["labels"], chart1["series"]))
        r3 = await ask(c, "Where does my money go? Show me a donut", thread)
        chart3 = (r3.get("charts") or [None])[0]
        check("a breakdown by category as a donut: at most 6 slices, the rest as Other",
              r3["status"] == "answered" and chart3 and chart3["form"] == "donut" and chart3["axis"] == "name"
              and 2 <= len(chart3["labels"]) <= 6 and "Groceries" in chart3["labels"])
        hist = (await c.get(f"/api/history?thread_id={thread}")).json()
        drawn = [m.get("charts") for m in hist["messages"] if m["role"] == "bank"]
        check("after a reload each chart is still under its answer",
              drawn == [r1["charts"], r2["charts"], r3["charts"]])

        # ---- the figures (transactions server) ----
        groceries = await tool(token, "transactions", "spending_by_month", {"category": "groceries"})
        summary = await tool(token, "transactions", "summarize_spending", {"category": "groceries"})
        today = date.today()
        first = date.fromisoformat(groceries["records_start"])
        expected, m = [], first.replace(day=1)
        while m <= today:
            expected.append(m.strftime("%Y-%m"))
            m = (m + timedelta(days=32)).replace(day=1)
        check("monthly figures: every month from where the records start to this month, none missing",
              groceries["months"] == expected and len(expected) >= 2)
        check("... the same total as the spending summary, and the chart drew exactly these",
              groceries["total"] == summary["totals"]["money_out"] > 0
              and chart1["series"][0]["values"] == groceries["series"][0]["by_month"])
        by_month, changes = groceries["series"][0]["by_month"], groceries["series"][0]["change_from_previous_month"]
        check("... each month compared with the one before, worked out by the bank (the AI never subtracts)",
              len(changes) == len(expected) - 1 and all(
                  c["month"] == expected[i + 1] and c["compared_with"] == expected[i]
                  and c["difference"] == abs(round(by_month[i + 1] - by_month[i], 2))
                  and c["more_or_less"] == ("more" if by_month[i + 1] > by_month[i] else
                                            "less" if by_month[i + 1] < by_month[i] else "the same")
                  for i, c in enumerate(changes))
              and "change_from_previous_month" not in json.dumps(chart1))
        check("... part months marked, and this month flagged as only so far",
              (expected[0] in groceries["partial"]) == (first.day != 1) and "this_month_so_far" in groceries
              and "so far" in groceries["partial"].get(expected[-1], "so far"))
        rows = GUEST_BANK.transactions(token, BANK_ID, GUEST_BANK.accounts_of(guest.guest_id)["savings"]) + \
            GUEST_BANK.transactions(token, BANK_ID, GUEST_BANK.accounts_of(guest.guest_id)["current"])
        own = [t for t in rows if t["details"]["description"] == "Monthly saving"]
        real_in = round(sum(float(t["details"]["value"]["amount"]) for t in rows
                            if float(t["details"]["value"]["amount"]) > 0
                            and t["details"]["description"] != "Monthly saving"), 2)
        income = await tool(token, "transactions", "spending_by_month", {"money": "in"})
        check("income by month leaves out money moved between her own accounts (the £200 monthly saving)",
              own and income["total"] == real_in and income["own_transfers_left_out"]["count"] == len(own) // 2)
        everything = await tool(token, "transactions", "summarize_spending", {})
        real_out = round(-sum(float(t["details"]["value"]["amount"]) for t in rows
                              if float(t["details"]["value"]["amount"]) < 0
                              and t["details"]["description"] != "Monthly saving"), 2)
        check("... and so does total spending", everything["totals"]["money_out"] == real_out
              and everything["own_transfers_left_out"]["count"] == len(own))
        older = await tool(token, "transactions", "spending_by_month",
                           {"category": "groceries", "from_date": (today - timedelta(days=400)).isoformat()})
        check("asked for more than the records hold: starts where they start, and says so",
              older["months"][0] == expected[0] and groceries["records_start"] in older["note"])
        both = await tool(token, "transactions", "spending_by_month", {"category": "groceries, eating out"})
        check("two categories at once: one line of figures each",
              [x["name"] for x in both["series"]] == ["Groceries", "Eating out"] and both["dataset_id"] != ds)
        backwards = await tool(token, "transactions", "spending_by_month",
                               {"from_date": "2026-10-01", "to_date": "2026-09-01"})
        check("dates the wrong way round are refused, and say so",
              backwards["error"] == "invalid_input" and "wrong way round" in backwards["message"])
        check("unknown categories and odd requests are refused plainly",
              (await tool(token, "transactions", "spending_by_month", {"category": "lottery"}))["error"] == "invalid_input"
              and (await tool(token, "transactions", "spending_by_month", {"money": "sideways"}))["error"] == "invalid_input"
              and (await tool(token, "transactions", "spending_by_month",
                              {"category": "groceries,coffee,rent,gym,bills"}))["error"] == "invalid_input"
              and (await tool(token, "transactions", "spending_by_month", {"account": "joint"}))["error"] == "access_denied")
        shops = await tool(token, "transactions", "spending_breakdown", {"by": "payee", "category": "groceries"})
        check("a breakdown of one category by shop: only grocers, biggest first",
              shops["items"] and all(categories_of(i["name"]) == "groceries" for i in shops["items"])
              and [i["total"] for i in shops["items"]] == sorted((i["total"] for i in shops["items"]), reverse=True))

        # ---- day by day, and this month so far against the same days of last month ----
        def money_out(r):
            return r["totals"]["money_out"]
        month1 = today.replace(day=1)
        daily = await tool(token, "transactions", "spending_by_day", {"category": "groceries"})
        sum_month = await tool(token, "transactions", "summarize_spending",
                               {"category": "groceries", "from_date": month1.isoformat(), "to_date": today.isoformat()})
        check("spending_by_day: every day of this month so far, totals from the bank that match summarize_spending",
              daily["days"] == today.day and daily["period"] == {"from": month1.isoformat(), "to": today.isoformat()}
              and abs(daily["total"] - money_out(sum_month)) < 0.005
              and abs(sum(d["amount"] for d in daily["by_day"]) - daily["total"]) < 0.005
              and (not daily["total"] or daily["highest_day"]["amount"] == max(d["amount"] for d in daily["by_day"])))
        too_long = await tool(token, "transactions", "spending_by_day",
                              {"from_date": (today - timedelta(days=90)).isoformat()})
        check("...at most 62 days, and it says to use month-by-month figures instead",
              too_long["error"] == "invalid_input" and "spending_by_month" in too_long["message"])
        day_cols = await tool(token, "charts", "column_chart", {"dataset_id": daily["dataset_id"]})
        drawn_day = charts.check(day_cols.get("chart")) if "chart" in day_cols else None
        check("...drawn as columns, one per day, which the page's checks accept",
              (drawn_day is not None and drawn_day["axis"] == "day" and len(drawn_day["labels"]) == today.day
               and abs(drawn_day["total"] - daily["total"]) < 0.005)
              or (today.day == 1 and day_cols.get("error") == "too_little"))
        check("...and a day-by-day chart can't be stacked or drawn as a donut",
              (await tool(token, "charts", "stacked_column_chart", {"dataset_id": daily["dataset_id"]}))["error"] == "wrong_form"
              and (await tool(token, "charts", "donut_chart", {"dataset_id": daily["dataset_id"]}))["error"] == "wrong_form")

        mtd = await tool(token, "transactions", "month_to_date", {"category": "groceries"})
        prev_last = month1 - timedelta(days=1)
        prev1, prev_end = prev_last.replace(day=1), prev_last.replace(day=min(today.day, prev_last.day))
        sum_prev = await tool(token, "transactions", "summarize_spending",
                              {"category": "groceries", "from_date": prev1.isoformat(), "to_date": prev_end.isoformat()})
        diff = round(money_out(sum_month) - money_out(sum_prev), 2)
        check(f"month_to_date: 1-{today.day} of this month against 1-{prev_end.day} of last month, from the bank",
              mtd["this_month"] == {"from": month1.isoformat(), "to": today.isoformat(),
                                    "total": money_out(sum_month), "payments": mtd["this_month"]["payments"]}
              and mtd["same_days_last_month"]["from"] == prev1.isoformat()
              and mtd["same_days_last_month"]["to"] == prev_end.isoformat()
              and abs(mtd["same_days_last_month"]["total"] - money_out(sum_prev)) < 0.005)
        check("...with the difference and percent worked out in code, and a sentence to quote",
              mtd["change"]["difference"] == abs(diff)
              and mtd["change"]["more_or_less"] == ("more" if diff > 0 else "less" if diff < 0 else "the same")
              and (mtd["change"]["percent"] == (round(abs(diff) / money_out(sum_prev) * 100, 1) if money_out(sum_prev) else None))
              and f"£{money_out(sum_month):,.2f}" in mtd["in_words"] and f"£{money_out(sum_prev):,.2f}" in mtd["in_words"])
        pace = await tool(token, "charts", "line_chart", {"dataset_id": mtd["dataset_id"]})
        drawn_pace = charts.check(pace.get("chart")) if "chart" in pace else None
        check("...drawn as two running-total lines by day of the month, which the page's checks accept",
              (drawn_pace is not None and drawn_pace["axis"] == "dayofmonth" and drawn_pace["running"]
               and drawn_pace["labels"] == [str(d) for d in range(1, today.day + 1)]
               and abs(drawn_pace["total"] - money_out(sum_month)) < 0.005
               and abs(drawn_pace["series"][1]["values"][-1] - money_out(sum_prev)) < 0.005)
              or (today.day == 1 and pace.get("error") == "too_little"))
        check("...and only as a line or a combined chart (columns would stack up running totals)",
              (await tool(token, "charts", "column_chart", {"dataset_id": mtd["dataset_id"]}))["error"] == "wrong_form")
        both_c = await tool(token, "charts", "combo_chart", {"dataset_id": mtd["dataset_id"]})
        drawn_both = charts.check(both_c.get("chart")) if "chart" in both_c else None
        check("combined chart of month to date: a column for each day, this month's running total, "
              "last month's dashed, all from the bank",
              (drawn_both is not None and [s["kind"] for s in drawn_both["series"]] == ["bar", "line", "dashed"]
               and abs(sum(drawn_both["series"][0]["values"]) - money_out(sum_month)) < 0.005
               and drawn_both["series"][1]["values"][-1] == drawn_pace["series"][0]["values"][-1]
               and drawn_both["series"][2]["values"] == drawn_pace["series"][1]["values"]
               and abs(drawn_both["total"] - money_out(sum_month)) < 0.005)
              or (today.day == 1 and both_c.get("error") == "too_little"))
        check("...its summary gives both periods' totals first, so the answer can repeat them",
              today.day == 1 or ("Days 1 to" in both_c["summary"]
                                 and f"{money_out(sum_month):,.2f}" in both_c["summary"]
                                 and f"{money_out(sum_prev):,.2f}" in both_c["summary"]))
        months_c = await tool(token, "charts", "column_chart", {"dataset_id": groceries["dataset_id"]})
        check("monthly chart summary gives each month's figure (when there are only a few)",
              len(groceries["months"]) > 6 or all(f"{v:,.2f}" in months_c["summary"]
                                                  for v in groceries["series"][0]["by_month"]))
        day_c = await tool(token, "charts", "combo_chart", {"dataset_id": daily["dataset_id"]})
        drawn_dc = charts.check(day_c.get("chart")) if "chart" in day_c else None
        check("combined chart of day-by-day figures: columns plus their running total (no last month)",
              (drawn_dc is not None and [s["kind"] for s in drawn_dc["series"]] == ["bar", "line"]
               and abs(drawn_dc["series"][1]["values"][-1] - daily["total"]) < 0.01)
              or (today.day == 1 and day_c.get("error") == "too_little"))
        check("...and month-by-month figures can't be drawn as a combined chart",
              (await tool(token, "charts", "combo_chart", {"dataset_id": ds}))["error"] == "wrong_form")
        check("as_of in the future is refused plainly",
              (await tool(token, "transactions", "month_to_date",
                          {"as_of": (today + timedelta(days=3)).isoformat()}))["error"] == "invalid_input")

        # ---- the drawing (chart server) ----
        stacked = await tool(token, "charts", "stacked_column_chart", {"dataset_id": both["dataset_id"]})
        check("stacked columns of two categories", stacked["chart"]["form"] == "stacked"
              and len(stacked["chart"]["series"]) == 2 and charts.check(stacked["chart"]) is not None)
        wrong = await tool(token, "charts", "donut_chart", {"dataset_id": ds})
        check("a donut of month-by-month figures doesn't fit: it says which forms do",
              wrong["error"] == "wrong_form" and "line_chart" in wrong["forms_that_fit"])
        single = await tool(token, "charts", "stacked_column_chart", {"dataset_id": ds})
        check("stacking a single category is refused", single["error"] == "wrong_form")
        one_month = await tool(token, "transactions", "spending_by_month",
                               {"category": "groceries", "from_date": today.replace(day=1).isoformat()})
        lone = await tool(token, "charts", "column_chart", {"dataset_id": one_month["dataset_id"]})
        figure = f"£{one_month['series'][0]['by_month'][0]:,.2f}"
        check(f"one month can't be charted, but the refusal carries its figure to give in words ({figure})",
              lone["error"] == "too_little" and figure in lone["message"] and "in words" in lone["message"])
        bars = await tool(token, "charts", "bar_chart", {"dataset_id": shops["dataset_id"], "top": 3})
        check("ranked bars of a breakdown, the rest as Other",
              bars["chart"]["labels"][-1] == "Other" and len(bars["chart"]["labels"]) == 4
              and round(sum(bars["chart"]["series"][0]["values"]), 2) == shops["total"])
        check("someone else's figures can't be drawn, even with the right id",
              (await tool("fake-token-A", "charts", "column_chart", {"dataset_id": ds}))["error"] == "not_found"
              and (await tool(token, "charts", "column_chart", {"dataset_id": "ds-00000000"}))["error"] == "not_found")
        await c.post("/api/guest/reset")
        check("Start over deletes the saved figures too",
              (await tool(token, "charts", "column_chart", {"dataset_id": ds}))["error"] == "not_found")

    good = {"form": "column", "axis": "month", "labels": ["2026-08", "2026-09"],
            "series": [{"name": "Groceries", "values": [1.5, 2.0], "counts": [1, 2]}]}
    odd = [{**good, "form": "pie"}, {**good, "labels": ["2026-08"]},
           {**good, "series": [{"name": "x", "values": [1.0, -2.0]}]},
           {**good, "series": [{"name": "x", "values": [1.0]}]}, {**good, "form": "donut"},
           {**good, "labels": ["2026-13", "2026-09"]}]
    check("charts: drawn only from chart tools, never from an error or odd data",
          charts.check(good) is not None and all(charts.check(x) is None for x in odd)
          and charts.from_tool("spending_by_month", json.dumps({"chart": good})) is None
          and charts.from_tool("column_chart", '{"error": "not_found"}') is None
          and charts.from_tool("column_chart", "not json") is None)
    day = {"form": "column", "axis": "day", "labels": ["2026-10-01", "2026-10-02"],
           "series": [{"name": "Groceries", "values": [0.0, 12.5]}]}
    run = {"form": "line", "axis": "dayofmonth", "running": True, "labels": ["1", "2", "3"],
           "series": [{"name": "October 2026", "values": [5.0, 5.0, 9.0]},
                      {"name": "September 2026", "values": [1.0, 4.0, 4.0]}]}
    odd_days = [{**day, "labels": ["2026-10", "2026-11"]}, {**day, "form": "stacked"}, {**run, "form": "column"},
                {**run, "labels": ["2", "3", "4"]},
                {**run, "series": [{"name": "x", "values": [5.0, 3.0, 9.0]}]}]   # a running total can't go down
    checked_run = charts.check(run)
    check("charts by day and running totals: drawn when sound, refused when odd",
          charts.check(day) is not None and checked_run is not None and checked_run["running"]
          and checked_run["total"] == 9.0 and all(charts.check(x) is None for x in odd_days))
    combo = {"form": "combo", "axis": "dayofmonth", "labels": ["1", "2", "3"],
             "series": [{"name": "Oct, each day", "kind": "bar", "values": [5.0, 0.0, 4.0]},
                        {"name": "Oct so far", "kind": "line", "values": [5.0, 5.0, 9.0]},
                        {"name": "Sep so far", "kind": "dashed", "values": [1.0, 4.0, 4.0]}]}
    odd_combos = [{**combo, "series": combo["series"][1:]},                                   # no columns
                  {**combo, "series": [combo["series"][1], combo["series"][0]]},              # columns not first
                  {**combo, "series": [combo["series"][0], {**combo["series"][1], "values": [5.0, 3.0, 9.0]}]},
                  {**combo, "axis": "month", "labels": ["2026-08", "2026-09", "2026-10"]},
                  {**combo, "series": [combo["series"][0]]}]                                  # columns alone
    checked_combo = charts.check(combo)
    check("combined charts: columns then running-total lines, total = the columns' sum; odd ones refused",
          checked_combo is not None and checked_combo["total"] == 9.0
          and [s["kind"] for s in checked_combo["series"]] == ["bar", "line", "dashed"]
          and charts.from_tool("combo_chart", json.dumps({"chart": combo})) is not None
          and all(charts.check(x) is None for x in odd_combos))


def categories_of(description: str) -> str | None:
    from obp.categories import category_of
    return category_of(description, money_in=False)


async def shutdown() -> None:
    """A copy stopping (scale-in, deploy) while answering: the answer is marked, not left 'running'."""
    app3 = api_module.create_app(model_factory=lambda: Stuck(messages=iter([])))
    async with client(app3) as c:
        async with app3.router.lifespan_context(app3):
            await c.post("/api/login", json={"email": "stuck@example.org"})
            r = await c.post("/api/chat", json={"message": "What are my balances?"})
            await asyncio.sleep(0.5)
        found = state.get_turn(r.json()["turn_id"])
        check("a copy stopping mid-answer marks the answer as interrupted",
              found["status"] == "failed" and "restarted" in found["result"]["error"])


class PretendGoogle:
    """Stands in for Google's token endpoint: checks what the app sends, answers with an ID token."""

    def __init__(self):
        self.claims: dict = {}
        self.status = 200
        self.seen: dict = {}
        self.challenge = ""

    def handler(self, request: httpx.Request) -> httpx.Response:
        from urllib.parse import parse_qsl
        self.seen = dict(parse_qsl(request.content.decode()))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "invalid_grant"})
        b64 = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()  # noqa: E731
        return httpx.Response(200, json={"id_token": f"{b64({'alg': 'RS256'})}.{b64(self.claims)}.sig",
                                         "access_token": "x"})


async def google_sign_in(c, fake: PretendGoogle, email: str, **change) -> httpx.Response:
    """Click "Continue with Google", come back from Google signed in as `email`."""
    from urllib.parse import parse_qs, urlsplit
    start = await c.get("/api/auth/google")
    q = {k: v[0] for k, v in parse_qs(urlsplit(start.headers["location"]).query).items()}
    fake.challenge = q["code_challenge"]
    fake.claims = {"iss": "https://accounts.google.com", "aud": "test-client.apps.googleusercontent.com",
                   "exp": time.time() + 3600, "nonce": q["nonce"], "email": email, "email_verified": True,
                   "given_name": email.split("@")[0].capitalize(), **change.pop("claims", {})}
    return await c.get("/api/auth/google/callback", params={"code": "one-time-code", "state": q["state"], **change})


async def verify_start(token: str, tool: str, args: dict) -> dict:
    async with httpx.AsyncClient(base_url=os.environ["VERIFY_URL"]) as v:
        r = await v.post("/verify/start", json={"tool": tool, "args": args},
                         headers={"Authorization": f'DirectLogin token="{token}"'})
        return r.json()


async def real_sign_in(app, owner_thread: str) -> None:
    """AUTH_MODE=public: Google, an emailed code, and the owner by email address."""
    import hashlib
    from urllib.parse import parse_qs, urlsplit

    from obp.guest_bank import token_for
    fake = PretendGoogle()
    app.state.google_transport = httpx.MockTransport(fake.handler)
    os.environ.update({"AUTH_MODE": "public", "PUBLIC_URL": "http://bank.test", "OWNER_EMAILS": "Owner.Krish@example.org",
                       "GOOGLE_CLIENT_ID": "test-client.apps.googleusercontent.com", "GOOGLE_CLIENT_SECRET": "test-secret"})
    async with client(app, ip="192.0.2.10") as c:
        cfg = (await c.get("/api/config")).json()
        check("public: Google and email offered, the owner's customers not listed to strangers",
              cfg["auth_mode"] == "public" and cfg["signin"] == {"google": True, "email": True, "invite_only": False} and cfg["customers"] == [])
        check("public: the development sign-in is off",
              (await c.post("/api/login", json={"email": "x@example.org"})).status_code == 404
              and (await c.post("/api/login", json={"customer": "A"})).status_code == 404)

        start = await c.get("/api/auth/google")
        url = urlsplit(start.headers["location"])
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        oauth_cookie = start.headers.get("set-cookie", "").lower()
        check("Google: sent to Google with our client, our return address, state, nonce and PKCE",
              start.status_code == 303 and url.netloc == "accounts.google.com"
              and q["client_id"] == "test-client.apps.googleusercontent.com"
              and q["redirect_uri"] == "http://bank.test/api/auth/google/callback"
              and q["code_challenge_method"] == "S256" and q["state"] and q["nonce"] and q["scope"] == "openid email profile")
        check("... remembering the attempt in a short-lived HttpOnly cookie",
              "httponly" in oauth_cookie and "path=/api/auth" in oauth_cookie and "samesite=lax" in oauth_cookie)
        # On AWS the app is reached through CloudFront and API Gateway, so the address it sees (http, an
        # internal host) is never PUBLIC_URL. It must still send people straight to Google (a redirect to
        # itself here looped forever on the first AWS deploy).
        os.environ["PUBLIC_URL"] = "https://d1234abcd.cloudfront.net"
        behind_proxy = await c.get("/api/auth/google")
        proxied = {k: v[0] for k, v in parse_qs(urlsplit(behind_proxy.headers["location"]).query).items()}
        check("behind CloudFront: straight to Google, with the public https return address (no redirect loop)",
              urlsplit(behind_proxy.headers["location"]).netloc == "accounts.google.com"
              and proxied["redirect_uri"] == "https://d1234abcd.cloudfront.net/api/auth/google/callback")
        os.environ["PUBLIC_URL"] = "http://127.0.0.1:8000"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000") as local:
            hop = await local.get("/api/auth/google")
        check("on your PC: opened as localhost, sent to 127.0.0.1 first (where Google returns)",
              hop.headers["location"] == "http://127.0.0.1:8000/api/auth/google")
        os.environ["PUBLIC_URL"] = "http://bank.test"
        c.cookies.clear()
        await c.get("/api/auth/google")   # a real start in this browser, then a return it didn't start
        r = await c.get("/api/auth/google/callback", params={"code": "x", "state": "made-up"})
        check("Google: a return that this browser didn't start is refused",
              r.headers["location"] == "/?signin_error=state" and (await c.get("/api/me")).status_code == 401)

        r = await google_sign_in(c, fake, "Gina@Example.org")
        verifier_ok = base64.urlsafe_b64encode(hashlib.sha256(fake.seen["code_verifier"].encode()).digest()
                                               ).rstrip(b"=").decode() == fake.challenge
        me = (await c.get("/api/me")).json()
        check("Google: signed in, back on the page, with a pretend bank whose address is proven",
              r.status_code == 303 and r.headers["location"] == "/" and me["kind"] == "guest"
              and me["email"] == "gina@example.org" and me["verified"] is True and me["name"] == "Gina")
        check("... the app swapped the code with its secret, PKCE verifier and return address",
              fake.seen["client_secret"] == "test-secret" and fake.seen["code_verifier"]
              and fake.seen["redirect_uri"] == "http://bank.test/api/auth/google/callback" and verifier_ok)
        gina = GUEST_BANK.active("gina@example.org")
        sent = await verify_start(token_for(gina.guest_id), "change_address", {"new_address": "2 Park Row, Leeds LS1 5HD"})
        check("security codes now go to the address she signed in with",
              sent["required"] and sent["sent_to"] == "g***@example.org" and outbox()[-1]["to"] == "gina@example.org")

    async with client(app, ip="192.0.2.11") as c:
        r = await google_sign_in(c, fake, "nina@example.org", claims={"email_verified": False})
        check("Google: an unverified Google address is refused", r.headers["location"] == "/?signin_error=unverified")
        r = await google_sign_in(c, fake, "nina@example.org", claims={"nonce": "replayed"})
        check("Google: a token for another sign-in (wrong nonce) is refused", r.headers["location"] == "/?signin_error=google")
        r = await google_sign_in(c, fake, "nina@example.org", claims={"aud": "someone-elses-app"})
        check("Google: a token for another app is refused", r.headers["location"] == "/?signin_error=google")
        fake.status = 400
        r = await google_sign_in(c, fake, "nina@example.org")
        fake.status = 200
        check("Google: a code Google won't accept is refused", r.headers["location"] == "/?signin_error=google")
        r = await c.get("/api/auth/google/callback", params={"error": "access_denied"})
        check("Google: pressing Cancel at Google comes back with a message",
              r.headers["location"] == "/?signin_error=cancelled" and (await c.get("/api/me")).status_code == 401)

        r = await google_sign_in(c, fake, "owner.krish@example.org")
        me = (await c.get("/api/me")).json()
        check("the owner (OWNER_EMAILS) signs in with Google and is asked to pick Robert or Anil",
              me["kind"] == "owner" and me["choosing"] is True and [x["name"] for x in me["customers"]] == ["Robert", "Anil"])
        check("... can't chat until they pick", (await c.post("/api/chat", json={"message": "hi"})).status_code == 409)
        r = await c.post("/api/owner/customer", json={"customer": "A"})
        me = (await c.get("/api/me")).json()
        check("... picks Robert, and finds Robert's conversations",
              r.status_code == 200 and me["customer"] == "A" and not me["choosing"]
              and (await c.get(f"/api/history?thread_id={owner_thread}")).status_code == 200)
        await c.post("/api/owner/customer", json={"customer": "B"})
        check("... can switch to Anil", (await c.get("/api/me")).json()["customer"] == "B"
              and (await c.get(f"/api/history?thread_id={owner_thread}")).status_code == 404)

    async with client(app, ip="192.0.2.12") as c:
        r = await c.post("/api/auth/email/start", json={"email": "Eve@Example.org"})
        sent = r.json()
        mail = outbox()[-1]
        code = re.search(r"sign-in code is (\d{6})", mail["body"]).group(1)
        check("email: a sign-in code is sent to the address", r.status_code == 200 and sent["sent_to"] == "e***@example.org"
              and mail["to"] == "eve@example.org" and mail["subject"] == "Your Demo Bank sign-in code")
        with storage.WEB.connect() as db:
            stored = " ".join(str(v) for row in db.execute("SELECT * FROM signin_codes").fetchall() for v in row)
        check("... and kept only as a hash", code not in stored)
        r = await c.post("/api/auth/email/check", json={"challenge": sent["challenge"], "code": "000000"})
        check("email: a wrong code shows the tries left", r.status_code == 400 and r.json() == {"error": "wrong_code", "attempts_left": 2})
        check("... and doesn't sign in", (await c.get("/api/me")).status_code == 401)
        r = await c.post("/api/auth/email/check", json={"challenge": sent["challenge"], "code": code})
        me = (await c.get("/api/me")).json()
        check("email: the right code signs in, with a proven address", r.status_code == 200
              and me["kind"] == "guest" and me["email"] == "eve@example.org" and me["verified"] is True)
        r = await c.post("/api/auth/email/check", json={"challenge": sent["challenge"], "code": code})
        check("email: a code works once", r.status_code == 400 and r.json()["error"] == "expired")
        for _ in range(2):
            await c.post("/api/auth/email/start", json={"email": "eve@example.org"})
        r = await c.post("/api/auth/email/start", json={"email": "eve@example.org"})
        check("email: at most 3 codes per address per 15 minutes (nobody's inbox gets flooded)",
              r.status_code == 429 and r.json()["error"] == "limit")
        os.environ["SIGNIN_CODES_PER_IP_HOUR"] = "3"   # this network has had 3
        r = await c.post("/api/auth/email/start", json={"email": "someone.else@example.org"})
        check("email: codes per internet address are limited", r.status_code == 429)
        os.environ["SIGNIN_CODES_PER_IP_HOUR"] = "10"
        os.environ["SIGNIN_EMAILS_PER_DAY"] = "1"
        async with client(app, ip="192.0.2.13") as d:
            r = await d.post("/api/auth/email/start", json={"email": "fresh@example.org"})
        check("email: a daily total keeps within Gmail's allowance", r.status_code == 429 and "Google" in r.json()["detail"])
        os.environ["SIGNIN_EMAILS_PER_DAY"] = "300"

    async with client(app, ip="192.0.2.14") as c:
        sent = (await c.post("/api/auth/email/start", json={"email": "owner.krish@example.org"})).json()
        code = re.search(r"sign-in code is (\d{6})", outbox()[-1]["body"]).group(1)
        await c.post("/api/auth/email/check", json={"challenge": sent["challenge"], "code": code})
        me = (await c.get("/api/me")).json()
        check("the owner can sign in with an email code too", me["kind"] == "owner" and me["choosing"])
    async with client(app, ip="192.0.2.15") as c:
        await google_sign_in(c, fake, "gina@example.org")
        check("a visitor can't make themselves a demo customer",
              (await c.post("/api/owner/customer", json={"customer": "A"})).status_code == 404)

    # Invite-only (the AWS preview): only the owner and invited addresses get in.
    os.environ.update({"OPEN_TO_PUBLIC": "0", "INVITED_EMAILS": "friend@example.org"})
    async with client(app, ip="192.0.2.16") as c:
        cfg = (await c.get("/api/config")).json()
        r = await c.post("/api/auth/email/start", json={"email": "stranger@example.org"})
        sent_before = len(outbox())
        check("invite-only: the page says so, and strangers aren't even sent a code",
              cfg["signin"]["invite_only"] and r.status_code == 403 and r.json()["error"] == "invite"
              and len(outbox()) == sent_before)
        check("invite-only: an invited address gets a code",
              (await c.post("/api/auth/email/start", json={"email": "friend@example.org"})).status_code == 200)
        r = await google_sign_in(c, fake, "stranger@example.org")
        check("invite-only: a stranger signing in with Google is turned away",
              r.headers["location"] == "/?signin_error=invite" and (await c.get("/api/me")).status_code == 401)
        await google_sign_in(c, fake, "owner.krish@example.org")
        check("invite-only: the owner still gets in", (await c.get("/api/me")).json()["kind"] == "owner")
    os.environ.update({"OPEN_TO_PUBLIC": "1", "INVITED_EMAILS": ""})

    os.environ.update({"AUTH_MODE": "dev", "OWNER_EMAILS": ""})


async def only_through_cloudfront(app) -> None:
    """On AWS, requests that skip CloudFront (straight to API Gateway's address) are turned away."""
    os.environ["ORIGIN_SECRET"] = "cloudfront-adds-this"
    async with client(app) as direct, httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://bank.test",
            headers={"X-Origin-Verify": "cloudfront-adds-this"}) as via_cloudfront:
        check("requests without CloudFront's secret header are refused",
              (await direct.get("/api/config")).status_code == 404 and (await direct.get("/")).status_code == 404)
        check("... the container's own health check still works", (await direct.get("/healthz")).status_code == 200)
        check("... and requests through CloudFront work", (await via_cloudfront.get("/api/config")).status_code == 200)
    os.environ["ORIGIN_SECRET"] = ""

async def main() -> int:
    print(f"Records: {storage.describe()}")
    app = api_module.create_app(model_factory=model)
    async with app.router.lifespan_context(app):
        owner_thread = await owner_flow(app)
        await visitor_flow(app, owner_thread)
        await verified_visitor_and_expiry(app)
        await chart_flow(app)
        await real_sign_in(app, owner_thread)
        await only_through_cloudfront(app)
        async with client(app) as a:
            await a.post("/api/login", json={"customer": "A"})
            await a.post("/api/logout")
            check("signed out: /api/me is 401", (await a.get("/api/me")).status_code == 401)
    await shutdown()
    left = list(SCRIPT)
    check("the scripted model was used exactly as planned", not left)
    print("\nALL API CHECKS PASSED" if OK else "\nSOME CHECKS FAILED")
    return 0 if OK else 1


if __name__ == "__main__":
    servers = harness.start_servers()
    try:
        code = asyncio.run(main(), loop_factory=storage.loop_factory())
    finally:
        harness.stop_servers(servers)
        storage._close_pools()
        if DROP_PG:
            DROP_PG()
    sys.exit(code)
