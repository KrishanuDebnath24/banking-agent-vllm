"""Offline end-to-end test of the phase 2 system.

Starts the real MCP servers and the security code service against the built-in
fake bank (OBP_FAKE=1), builds the real coordinator + subagents, and drives
them with a scripted fake model. Codes are "emailed" to a file the test reads.
No API keys or network needed.

    python tests/test_multi_agent_offline.py
"""

from __future__ import annotations

import asyncio
import builtins
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="bank-test-"))
DEMO = TMP / "demo_accounts.json"
OUTBOX = TMP / "outbox.jsonl"
DEMO.write_text(json.dumps({"bank_id": "gh.29.uk", "customers": {
    "A": {"username": "robert.uk.29@example.com", "currency": "GBP", "current": "a-cur", "savings": "a-sav",
          "income": "a-inc", "merchants": "a-mer", "funding": []},
    "B": {"username": "anil.uk.29@example.com", "currency": "GBP", "current": "b-cur", "savings": "b-sav",
          "income": "b-inc", "merchants": "b-mer", "funding": []}}}))
TEST_ENV = {"BANK_TEST": "1", "TELEMETRY_DB": "off", "MCP_PORT_OFFSET": "200",  # own ports: 8301-8305
            "MCP_ACCOUNTS_URL": "http://127.0.0.1:8301/mcp", "MCP_TRANSACTIONS_URL": "http://127.0.0.1:8302/mcp",
            "MCP_SERVICE_URL": "http://127.0.0.1:8303/mcp", "VERIFY_URL": "http://127.0.0.1:8304",
            "MCP_CHARTS_URL": "http://127.0.0.1:8305/mcp", "CHART_DB": str(TMP / "charts.db"),
            "OBP_FAKE": "1", "DEMO_ACCOUNTS_FILE": str(DEMO),
            "SERVICE_REQUESTS_FILE": str(TMP / "requests.jsonl"), "SERVICE_REQUESTS_DB": str(TMP / "requests.db"),
            "DATABASE_URL": "", "LANGSMITH_TRACING": "false",
            "EMAIL_BACKEND": "console", "OTP_OUTBOX": str(OUTBOX), "OTP_DB": str(TMP / "verification.db"),
            "OTP_TRANSFER_THRESHOLD": "100", "OTP_EMAIL_TO": "me@demo.test"}
ENV = {**os.environ, **TEST_ENV}
os.environ.update(TEST_ENV)

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agents.coordinator import build_coordinator, load_mcp_tools  # noqa: E402
from observability import costs  # noqa: E402
from scripts.chat_multi import CODE_NOT_CONFIRMED, security_code  # noqa: E402


class Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def service_requests() -> list[dict]:
    """Every service request the servers saved, oldest first (from the service_requests table)."""
    import storage
    with storage.REQUESTS.connect() as db:
        rows = db.execute("SELECT reference, type, status, bank_message FROM service_requests"
                          " ORDER BY created_ts").fetchall()
    return [dict(zip(("reference", "type", "status", "bank_message"), r)) for r in rows]


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "id": f"id-{uuid.uuid4().hex}", "args": args}])


def priced(message):
    """Attach token usage like a real Claude response, for the cost meter."""
    message.usage_metadata = {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050}
    message.response_metadata = {"model_name": "claude-haiku-5-5"}
    return message


def task(agent, description):
    return call("task", subagent_type=agent, description=description)


def latest_code() -> str | None:
    if not OUTBOX.exists():
        return None
    lines = OUTBOX.read_text().splitlines()
    m = re.search(r"security code is (\d{6})", json.loads(lines[-1])["body"]) if lines else None
    return m.group(1) if m else None


def emails() -> int:
    return len(OUTBOX.read_text().splitlines()) if OUTBOX.exists() else 0


def keyboard(*answers):
    """Stand-in for the customer typing at the 'Security code:' prompt. "CODE" = the emailed code."""
    queue = list(answers)

    def fake_input(prompt=""):
        if not queue:
            return ""  # Enter = cancel
        answer = queue.pop(0)
        return latest_code() if answer == "CODE" else answer
    builtins.input = fake_input


def contains_code(code: str, data: bytes) -> bool:
    """The code as a standalone number (not part of a longer number or hex id)."""
    return re.search(rb"(?<![0-9a-f])" + code.encode() + rb"(?![0-9a-f])", data) is not None


async def tool(token, server, name, **args):
    tools = {t.name: t for t in (await load_mcp_tools(token))[server]}
    out = await tools[name].ainvoke(args)
    return json.loads(out[0]["text"] if isinstance(out, list) else out)


async def balances(token="fake-token-A"):
    return {a["account"]: a["balance"] for a in await tool(token, "accounts", "list_my_accounts")}


async def scenario(db, thread, script, user_text, decide=None, token="fake-token-A", meter=None):
    """Run one turn. `decide(action)` returns the decision for a paused action, like the chat app."""
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        agent = await build_coordinator(token, saver, model=Fake(messages=iter(script)), bedrock=False)
        cfg = {"configurable": {"thread_id": thread}}
        if meter is not None:
            cfg["callbacks"] = [costs.CostCallback(meter)]
        await agent.ainvoke({"messages": [{"role": "user", "content": user_text}]}, cfg)
        state = await agent.aget_state(cfg)
        actions = [a for i in state.interrupts for a in i.value["action_requests"]]
        if decide and state.interrupts:
            await agent.ainvoke(Command(resume={"decisions": [await decide(a) for a in actions]}), cfg)
            state = await agent.aget_state(cfg)
        return [a["name"] for a in actions], state


def approve_only():
    async def decide(action):
        return {"type": "approve"}      # a client that skips the security code step
    return decide


def approve_with_code(token="fake-token-A"):
    async def decide(action):           # what chat_multi does after "y"
        ok = await security_code(token, action)
        return {"type": "approve"} if ok else {"type": "reject", "message": CODE_NOT_CONFIRMED}
    return decide


async def main() -> int:
    db = TMP / "sessions.db"
    ok = True

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + label)

    start = await balances()
    print("start balances:", start)

    # 1. Small transfer: pauses for y/N, needs no code, moves money. Also checks the cost meter.
    sent_before = emails()
    meter = costs.start_turn("A-t1", "A")
    pending, st = await scenario(db, "A-t1", [
        priced(task("accounts-agent", "Transfer £50.00 from current to savings.")),
        priced(call("transfer_between_my_accounts", from_account="current", to_account="savings", amount=50)),
        priced(AIMessage(content="Transferred £50.00.")), priced(AIMessage(content="Done."))],
        "move £50 to savings", approve_with_code(), meter=meter)
    costs.end_turn(meter, save=False)
    agents_seen = {c["agent"] for c in meter.llm_calls}
    check("cost meter sees all 4 model calls, coordinator and subagent",
          len(meter.llm_calls) == 4 and "accounts-agent" in agents_seen and len(agents_seen) == 2)
    check("cost meter prices them", abs(meter.llm_cost - 4 * (1000 * 0.10 + 50 * 0.50) / 1e6) < 1e-9)
    print("   agents seen by the cost meter:", sorted(agents_seen))
    after = await balances()
    check("transfer pauses for approval", pending == ["transfer_between_my_accounts"])
    check("£50 transfer needs no security code", emails() == sent_before)
    check("approved transfer moves £50", round(after["savings"] - start["savings"], 2) == 50.0)

    # 2. Transfer rejected: nothing moves.
    pending, st = await scenario(db, "A-t2", [
        task("accounts-agent", "Transfer £25.00 from current to savings."),
        call("transfer_between_my_accounts", from_account="current", to_account="savings", amount=25),
        AIMessage(content="The customer declined; nothing was done."), AIMessage(content="No problem, nothing moved.")],
        "move £25 to savings", lambda a: asyncio.sleep(0, {"type": "reject", "message": "Declined."}))
    check("rejected transfer moves nothing", await balances() == after)

    # 3. Large transfer approved WITHOUT a code: the server itself refuses.
    pending, st = await scenario(db, "A-t3", [
        task("accounts-agent", "Transfer £600.00 from current to savings."),
        call("transfer_between_my_accounts", from_account="current", to_account="savings", amount=600),
        AIMessage(content="Not done: code not confirmed."), AIMessage(content="That wasn't done.")],
        "move £600 to savings", approve_only())
    direct = await tool("fake-token-A", "accounts", "transfer_between_my_accounts",
                        from_account="current", to_account="savings", amount=600)
    check("£600 without a code is refused by the server", direct.get("error") == "verification_required")
    check("refused £600 moves nothing", await balances() == after)

    # 4. Large transfer with the emailed code (one wrong try first): money moves.
    keyboard("999999x", "CODE")
    pending, st = await scenario(db, "A-t4", [
        task("accounts-agent", "Transfer £600.00 from current to savings."),
        call("transfer_between_my_accounts", from_account="current", to_account="savings", amount=600),
        AIMessage(content="Transferred £600.00."), AIMessage(content="Done, £600 moved.")],
        "move £600 to savings", approve_with_code())
    code = latest_code()
    after4 = await balances()
    check("£600 code was emailed and shows the amount",
          code and "Transfer £600.00 from current to savings" in json.loads(OUTBOX.read_text().splitlines()[-1])["body"])
    check("£600 with the right code moves money", round(after4["savings"] - after["savings"], 2) == 600.0)
    conversation = json.dumps([str(m.content) for m in st.values["messages"]] +
                              [str(getattr(m, "tool_calls", "")) for m in st.values["messages"]])
    check("the code never reaches the model or the conversation", not contains_code(code, conversation.encode()))
    check("the code is not stored in the session store",
          not any(contains_code(code, f.read_bytes()) for f in TMP.glob("sessions.db*")))

    # 5. A code only works for the exact action it was sent for, and only once.
    keyboard("CODE")
    approved = await security_code("fake-token-A", {"name": "transfer_between_my_accounts", "args": {
        "from_account": "savings", "to_account": "current", "amount": 200}})
    other = await tool("fake-token-A", "accounts", "transfer_between_my_accounts",
                       from_account="savings", to_account="current", amount=210)
    same = await tool("fake-token-A", "accounts", "transfer_between_my_accounts",
                      from_account="savings", to_account="current", amount=200)
    again = await tool("fake-token-A", "accounts", "transfer_between_my_accounts",
                       from_account="savings", to_account="current", amount=200)
    check("code confirmed for £200", approved)
    check("a £200 code can't approve £210", other.get("error") == "verification_required")
    check("the £200 code approves £200", same.get("status") == "COMPLETED")
    check("a code can't be used twice", again.get("error") == "verification_required")

    # 6. Three wrong codes kill the code, even if the right one comes next.
    keyboard("111111x", "222222x", "333333x")
    check("three wrong codes are refused", not await security_code(
        "fake-token-A", {"name": "change_address", "args": {"new_address": "1 Wrong Lane, Dublin D01 X2Y3"}}))

    # 7. Another customer can't use A's code.
    import httpx
    async with httpx.AsyncClient(base_url=os.environ["VERIFY_URL"]) as http:
        a = (await http.post("/verify/start", headers={"Authorization": 'DirectLogin token="fake-token-A"'},
                             json={"tool": "change_address", "args": {"new_address": "9 Elm Road, Cork T12 AB34"}})).json()
        b = (await http.post("/verify/check", headers={"Authorization": 'DirectLogin token="fake-token-B"'},
                             json={"challenge_id": a["challenge_id"], "code": latest_code()})).json()
        big = (await http.post("/verify/start", headers={"Authorization": 'DirectLogin token="fake-token-A"'},
                               json={"tool": "transfer_between_my_accounts",
                                     "args": {"from_account": "current", "to_account": "savings", "amount": 900}})).json()
    check("customer B can't use customer A's code", b.get("verified") is False)
    check("over-limit transfer asks for no code (the bank refuses it anyway)", big == {"required": False})

    # 8. Address change: code, then saved at the bank and read back by the profile.
    new_address = "12 Main Street, Maynooth, W23 X1Y2"
    keyboard("CODE")
    pending, st = await scenario(db, "A-t8", [
        task("service-agent", f"Change the customer's address to {new_address}."),
        call("change_address", new_address=new_address),
        AIMessage(content="Address changed."), AIMessage(content="Your address is updated.")],
        "change my address", approve_with_code())
    profile = await tool("fake-token-A", "service", "get_my_profile")
    reqs = service_requests()
    check("address change pauses for approval", pending == ["change_address"])
    check("new address is saved at the bank (profile reads it back)", profile.get("address") == new_address)
    check("address change logged as completed with a reference", reqs[-1]["type"] == "address_change"
          and reqs[-1]["status"] == "completed" and reqs[-1]["reference"].startswith("SR-"))
    check("customer told about the change (bank message sent)", reqs[-1].get("bank_message") == "sent")
    other_profile = await tool("fake-token-B", "service", "get_my_profile")
    check("customer B's address is untouched", other_profile.get("address") != new_address)

    # 9. Address change without a code: refused, address unchanged.
    pending, st = await scenario(db, "A-t9", [
        task("service-agent", "Change the customer's address to 5 Fake Road, Galway H91 ZZ99."),
        call("change_address", new_address="5 Fake Road, Galway H91 ZZ99"),
        AIMessage(content="Not done."), AIMessage(content="That wasn't done.")],
        "change my address", approve_only())
    direct = await tool("fake-token-A", "service", "change_address", new_address="5 Fake Road, Galway H91 ZZ99")
    check("address change without a code is refused", direct.get("error") == "verification_required")
    check("address unchanged after the refused change",
          (await tool("fake-token-A", "service", "get_my_profile")).get("address") == new_address)

    # 10. Pending confirmation survives a restart (session store).
    pending, st = await scenario(db, "A-t10", [
        task("service-agent", "Order a cheque book for the current account."),
        call("request_cheque_book", account="current")], "cheque book please")
    check("cheque book pauses for approval", pending == ["request_cheque_book"])
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:  # "restart": new saver, new agent
        agent = await build_coordinator("fake-token-A", saver, bedrock=False, model=Fake(messages=iter([
            AIMessage(content="Cheque book ordered."), AIMessage(content="Your cheque book is on its way.")])))
        cfg = {"configurable": {"thread_id": "A-t10"}}
        state = await agent.aget_state(cfg)
        check("pending confirmation restored after restart", len(state.interrupts) == 1)
        await agent.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg)
        reqs = service_requests()
        check("resumed cheque book request completes", reqs[-1]["type"] == "cheque_book")

    # 11. Another customer's account is refused by the server itself, whatever the model asks.
    as_a = await tool("fake-token-A", "accounts", "get_balance", account="b-cur")   # A asking for B's account
    as_b = await tool("fake-token-B", "accounts", "get_balance", account="b-cur")   # B asking for their own
    bad = await tool("made-up-token", "accounts", "get_balance", account="current")
    check("server refuses A reading B's account", as_a.get("error") == "access_denied")
    check("server allows B reading B's own account", "balance" in as_b)
    check("server refuses an unknown token", bad.get("error") == "not_authorised")
    said = await tool("fake-token-A", "accounts", "get_balance", account="my Current Account")
    joint = await tool("fake-token-A", "accounts", "get_balance", account="joint account")
    check("'my Current Account' is understood as current", said.get("account") == "current")
    check("'joint account' is still refused", joint.get("error") == "access_denied")

    print("\nALL MULTI-AGENT CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    servers = subprocess.Popen([sys.executable, "-m", "mcp_servers.run"], cwd=ROOT, env=ENV,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # Wait until all five servers answer (a fixed pause was sometimes too short on a slow CI machine).
        import socket

        def answering(port: int) -> bool:
            with socket.socket() as sock:
                return sock.connect_ex(("127.0.0.1", port)) == 0

        ports, deadline = [8301, 8302, 8303, 8304, 8305], time.time() + 90
        while time.time() < deadline and servers.poll() is None:
            if all(answering(p) for p in ports):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("the bank servers didn't start (ports 8301-8305)")
        code = asyncio.run(main())
    finally:
        servers.terminate()
    sys.exit(code)
