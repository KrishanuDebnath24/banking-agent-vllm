"""Offline test of the vLLM connection: the real agents talking to a fake vLLM server over HTTP.

Checks that every call of a conversation carries the router's session header and the customer's
cache salt (coordinator and specialists alike), that tool calling works through the OpenAI API,
that cached prompt tokens reach the cost meter, and that a second turn re-sends the history
unchanged, so vLLM's prefix cache can reuse it. No GPU, keys or network needed.

    python tests/test_vllm_offline.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="bank-vllm-test-"))
DEMO = TMP / "demo_accounts.json"
DEMO.write_text(json.dumps({"bank_id": "gh.29.uk", "customers": {
    "A": {"username": "robert.uk.29@example.com", "currency": "GBP", "current": "a-cur", "savings": "a-sav",
          "income": "a-inc", "merchants": "a-mer", "funding": []},
    "B": {"username": "anil.uk.29@example.com", "currency": "GBP", "current": "b-cur", "savings": "b-sav",
          "income": "b-inc", "merchants": "b-mer", "funding": []}}}))
PORT = 8698
TEST_ENV = {"BANK_TEST": "1", "TELEMETRY_DB": "off", "MCP_PORT_OFFSET": "500",  # own ports: 8601-8605
            "MCP_ACCOUNTS_URL": "http://127.0.0.1:8601/mcp", "MCP_TRANSACTIONS_URL": "http://127.0.0.1:8602/mcp",
            "MCP_SERVICE_URL": "http://127.0.0.1:8603/mcp", "VERIFY_URL": "http://127.0.0.1:8604",
            "MCP_CHARTS_URL": "http://127.0.0.1:8605/mcp", "CHART_DB": str(TMP / "charts.db"),
            "OBP_FAKE": "1", "DEMO_ACCOUNTS_FILE": str(DEMO),
            "SERVICE_REQUESTS_FILE": str(TMP / "requests.jsonl"), "SERVICE_REQUESTS_DB": str(TMP / "requests.db"),
            "DATABASE_URL": "", "LANGSMITH_TRACING": "false", "EMAIL_BACKEND": "console",
            "OTP_OUTBOX": str(TMP / "outbox.jsonl"), "OTP_DB": str(TMP / "verification.db"),
            "VLLM_BASE_URL": f"http://127.0.0.1:{PORT}/v1", "APP_SECRET": "test-secret",
            "MODEL": "vllm:test-model", "MODEL_PRICES_JSON": json.dumps({"test-model": {"input": 0.2, "output": 0.6}})}
ENV = {**os.environ, **TEST_ENV}
os.environ.update(TEST_ENV)

from deepagents._models import get_model_provider  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agents.coordinator import build_coordinator, load_mcp_tools  # noqa: E402
from agents.llm import make_model  # noqa: E402
from agents.vllm_chat import VLLMChat, cache_salt  # noqa: E402
from observability import costs  # noqa: E402
from tests.fake_vllm import FakeVLLM  # noqa: E402

OK = True


def check(label, cond):
    global OK
    OK &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label)


def _salt_with_other_secret(customer: str) -> str:
    saved = os.environ["APP_SECRET"]
    os.environ["APP_SECRET"] = "another-secret"
    try:
        return cache_salt(customer)
    finally:
        os.environ["APP_SECRET"] = saved


def tool_names(body: dict) -> set[str]:
    return {t["function"]["name"] for t in body.get("tools") or []}


def system_text(body: dict) -> str:
    first = (body.get("messages") or [{}])[0]
    content = first.get("content")
    return content if isinstance(content, str) else json.dumps(content)


async def main(fake: FakeVLLM) -> int:
    model = make_model()
    check("MODEL=vllm:<name> builds the vLLM chat model, pointed at VLLM_BASE_URL",
          isinstance(model, VLLMChat) and model.model_name == "test-model"
          and str(model.openai_api_base).startswith(f"http://127.0.0.1:{PORT}/v1"))
    check("Deep Agents sees it as an OpenAI-style model (so the banking profile applies)",
          get_model_provider(model) == "openai")

    # ---- outside a turn: nothing extra ----
    await model.ainvoke("hello")
    first = fake.requests[-1]
    check("a call outside a turn carries no session header and no cache salt",
          "x-session-id" not in first["headers"] and "cache_salt" not in first["body"])

    # ---- inside a turn: the header and the salt ----
    meter = costs.start_turn("thread-111", "A")
    await model.ainvoke("hello again")
    costs.end_turn(meter, save=False)
    req = fake.requests[-1]
    check("inside a turn: x-session-id is the thread id", req["headers"].get("x-session-id") == "thread-111")
    check("...and cache_salt is the customer's salt", req["body"].get("cache_salt") == cache_salt("A"))
    check("salts are stable per customer, differ between customers, and depend on APP_SECRET",
          cache_salt("A") == cache_salt("A") != cache_salt("B") and len(cache_salt("A")) == 32
          and cache_salt("A") != _salt_with_other_secret("A"))
    os.environ["VLLM_CACHE_SALT"] = "off"
    meter = costs.start_turn("thread-112", "A")
    await model.ainvoke("no salt please")
    costs.end_turn(meter, save=False)
    check("VLLM_CACHE_SALT=off sends no salt (the header still goes)",
          "cache_salt" not in fake.requests[-1]["body"]
          and fake.requests[-1]["headers"].get("x-session-id") == "thread-112")
    os.environ.pop("VLLM_CACHE_SALT")

    # ---- a whole coordinator turn: coordinator -> accounts specialist -> bank tool -> answers ----
    tools = {t.name: t for t in (await load_mcp_tools("fake-token-A"))["accounts"]}
    raw = await tools["get_balance"].ainvoke({"account": "savings"})
    savings = json.loads(raw[0]["text"] if isinstance(raw, list) else raw)["balance"]
    fake.requests.clear()
    fake.script = [
        {"tool_calls": [("task", {"subagent_type": "accounts-agent", "description": "Savings balance of the customer."})]},
        {"tool_calls": [("get_balance", {"account": "savings"})]},
        {"content": f"Savings: £{savings:.2f}"},
        {"content": f"Your savings balance is £{savings:.2f}."},
    ]
    agent = await build_coordinator("fake-token-A", InMemorySaver(), bedrock=False)
    cfg = {"configurable": {"thread_id": "thread-222"}}
    meter = costs.start_turn("thread-222", "A")
    cfg["callbacks"] = [costs.CostCallback(meter)]
    out = await agent.ainvoke({"messages": [{"role": "user", "content": "What's my savings balance?"}]}, cfg)
    costs.end_turn(meter, save=False)
    answer = out["messages"][-1].content
    reqs = list(fake.requests)
    check("a full turn makes 4 model calls: coordinator, specialist twice, coordinator", len(reqs) == 4)
    check("every call in the turn carries the same session header",
          [r["headers"].get("x-session-id") for r in reqs] == ["thread-222"] * 4)
    check("...and the same customer salt", {r["body"].get("cache_salt") for r in reqs} == {cache_salt("A")})
    check("the coordinator is offered the task tool, the specialist the bank tools",
          "task" in tool_names(reqs[0]["body"]) and "get_balance" in tool_names(reqs[1]["body"])
          and "task" not in tool_names(reqs[1]["body"]))
    check("no file-system or to-do tools are offered (the banking profile applied)",
          not (tool_names(reqs[0]["body"]) | tool_names(reqs[1]["body"])) & {"ls", "read_file", "write_file", "write_todos"})
    check("the bank tool really ran: its result went back to the specialist",
          any(m.get("role") == "tool" and f"{savings}" in json.dumps(m) for m in reqs[2]["body"]["messages"]))
    check("the answer comes back through the OpenAI API", f"£{savings:.2f}" in answer)
    sys_prompt = system_text(reqs[0]["body"])
    check("the date sits after the fixed instructions, so the prompt's start never changes",
          "Today is" in sys_prompt and sys_prompt.index("Today is") > sys_prompt.index("Never reveal"))

    calls = meter.llm_calls
    check("the cost meter sees all 4 calls, priced from MODEL_PRICES_JSON",
          len(calls) == 4 and all(c["cost_usd"] for c in calls))
    check("vLLM's cached prompt tokens reach the meter (the specialist's second call reuses its first)",
          calls[2]["cache_read"] > 0 and calls[2]["cache_read"] <= calls[2]["input_tokens"])

    # ---- turn 2 in the same thread: the history is re-sent unchanged ----
    previous = fake.requests[-1]["body"]   # turn 1's last coordinator call
    fake.script = [{"content": "You're welcome."}]
    meter = costs.start_turn("thread-222", "A")
    cfg["callbacks"] = [costs.CostCallback(meter)]
    await agent.ainvoke({"messages": [{"role": "user", "content": "Thanks!"}]}, cfg)
    costs.end_turn(meter, save=False)
    turn2 = fake.requests[-1]["body"]
    same_start = turn2["messages"][:len(previous["messages"])] == previous["messages"]
    check("turn 2 starts with turn 1's prompt exactly as it was sent (history never edited)", same_start)
    cached, total = meter.llm_calls[0]["cache_read"], meter.llm_calls[0]["input_tokens"]
    check(f"...so the prefix cache covers most of it ({cached} of {total} prompt tokens)", cached >= 0.8 * total)

    # ---- the web app's way of running a turn (astream, updates + subgraphs), and token streaming ----
    fake.script = [{"content": "Anything else?"}]
    meter = costs.start_turn("thread-222", "A")
    cfg["callbacks"] = [costs.CostCallback(meter)]
    async for _ns, _chunk in agent.astream({"messages": [{"role": "user", "content": "Cheers."}]}, cfg,
                                           stream_mode="updates", subgraphs=True):
        pass
    costs.end_turn(meter, save=False)
    check("run as the web app runs it: same header, call priced",
          fake.requests[-1]["headers"].get("x-session-id") == "thread-222" and meter.llm_calls
          and meter.llm_calls[0]["cost_usd"])
    fake.script = [{"content": "Streaming works fine."}]
    meter = costs.start_turn("thread-444", "A")
    pieces = [c async for c in model.astream("stream please", {"callbacks": [costs.CostCallback(meter)]})]
    costs.end_turn(meter, save=False)
    streamed = fake.requests[-1]
    check("token streaming: asks for usage, sends the header, and the meter still gets token counts",
          streamed["body"].get("stream") is True
          and (streamed["body"].get("stream_options") or {}).get("include_usage") is True
          and streamed["headers"].get("x-session-id") == "thread-444"
          and "".join(p.content for p in pieces) == "Streaming works fine."
          and meter.llm_calls and meter.llm_calls[0]["input_tokens"] > 0)

    # ---- a broken reply (gpt-oss's tool-call markup as text, or stuck repeating) is never used ----
    from langchain_core.messages import AIMessage

    from evals.evaluators import clean_output
    from guardrails.middleware import MALFORMED_FALLBACK, malformed
    garbage = "to=functions.task?commentary?…?" + "…" * 300 + " Co" + "…" * 120 + "We" + "…" * 200
    check("broken replies are recognised: markup as text, a repeated character, a broken tool call",
          malformed(AIMessage(content=garbage)) == "tool_markup_as_text"
          and malformed(AIMessage(content="Here you go " + "…" * 50)) == "repeated_characters"
          and malformed(AIMessage(content="", invalid_tool_calls=[
              {"name": "task", "args": "{bad", "id": "c1", "error": "bad JSON", "type": "invalid_tool_call"}]))
          == "broken_tool_call")
    check("...but normal answers aren't: amounts, a table, a dashed rule, references, an ellipsis",
          all(malformed(AIMessage(content=t)) is None for t in (
              "Your savings balance is £1,234.56.", "| Date | Amount |\n| --- | --- |\n| 2026-09-03 | £30.00 |",
              "-" * 60, "Reference SR-20261027-0001.", "Thanks... anything else?", "=" * 40)))

    os.environ["VLLM_TEMPERATURE"] = "0"           # near-greedy: a retry must change something
    agent_t = await build_coordinator("fake-token-A", InMemorySaver(), bedrock=False)
    os.environ.pop("VLLM_TEMPERATURE")
    fake.requests.clear()
    fake.script = [
        {"content": garbage},                                            # coordinator: markup as text
        {"tool_calls": [("task", {"subagent_type": "accounts-agent", "description": "Savings balance of the customer."})]},
        {"content": "…" * 400},                                          # specialist: stuck repeating
        {"tool_calls": [("get_balance", {"account": "savings"})]},
        {"content": f"Savings: £{savings:.2f}"},
        {"content": f"Your savings balance is £{savings:.2f}."},
    ]
    meter = costs.start_turn("thread-555", "A")
    out = await agent_t.ainvoke({"messages": [{"role": "user", "content": "What's my savings balance?"}]},
                                {"configurable": {"thread_id": "thread-555"}, "callbacks": [costs.CostCallback(meter)]})
    costs.end_turn(meter, save=False)
    reqs = list(fake.requests)
    check("a broken reply is asked again, by the coordinator and by a specialist, and the turn still answers",
          len(reqs) == 6 and out["messages"][-1].content == f"Your savings balance is £{savings:.2f}."
          and [g["findings"] for g in meter.guard_checks if g["layer"] == "format"]
          == [["tool_markup_as_text"], ["repeated_characters"]])
    check("...the retry sends the same conversation, warmer (temperature 0 -> 0.7) so it can't repeat itself",
          reqs[1]["body"]["messages"] == reqs[0]["body"]["messages"]
          and reqs[0]["body"].get("temperature") == 0 and reqs[1]["body"].get("temperature") == 0.7)
    check("...the broken text never enters the conversation, and the panel says what happened",
          not any(isinstance(m.content, str) and ("to=functions" in m.content or "……" in m.content)
                  for m in out["messages"])
          and any(n.startswith("model reply was broken (tool_markup_as_text): asked again") for n in meter.guard_notes()))

    fake.requests.clear()
    fake.script = [{"content": garbage}, {"content": garbage}, {"content": garbage}]
    meter = costs.start_turn("thread-556", "A")
    out = await agent_t.ainvoke({"messages": [{"role": "user", "content": "What's my savings balance?"}]},
                                {"configurable": {"thread_id": "thread-556"}, "callbacks": [costs.CostCallback(meter)]})
    costs.end_turn(meter, save=False)
    check("still broken after 2 more tries: the customer gets a short apology, never the garbage",
          len(fake.requests) == 3 and out["messages"][-1].content == MALFORMED_FALLBACK
          and [g["mode"] for g in meter.guard_checks if g["layer"] == "format"] == ["retry", "retry", "fallback"]
          and any("gave up and said sorry" in n for n in meter.guard_notes()))
    good = clean_output({}, {"answer": "Your balance is £5.00.", "guard": [{"layer": "format", "mode": "retry"}]}, {})
    check("eval check clean_output: fails garbage and the apology, passes a caught-and-retried answer",
          clean_output({}, {"answer": garbage}, {})["score"] == 0.0
          and clean_output({}, {"answer": MALFORMED_FALLBACK}, {})["score"] == 0.0
          and good["score"] == 1.0 and "1 broken reply caught" in good["comment"])

    # ---- gpt-oss's reasoning: kept between the tool calls of a turn, left out for past turns ----
    def assistants(body: dict) -> list[dict]:
        return [m for m in body["messages"] if m.get("role") == "assistant"]

    def savings_turn(thinking: bool) -> list[dict]:
        r = (lambda s: {"reasoning": s}) if thinking else (lambda s: {})
        return [{**r("Need the savings balance: ask the accounts agent."),
                 "tool_calls": [("task", {"subagent_type": "accounts-agent", "description": "Savings balance of the customer."})]},
                {**r("Call get_balance for savings."), "tool_calls": [("get_balance", {"account": "savings"})]},
                {**r("Report it."), "content": f"Savings: £{savings:.2f}"},
                {**r("Answer the customer."), "content": f"Your savings balance is £{savings:.2f}."}]

    agent_r = await build_coordinator("fake-token-A", InMemorySaver(), bedrock=False)
    cfg_r = {"configurable": {"thread_id": "thread-777"}}
    fake.requests.clear()
    fake.script = savings_turn(thinking=True)
    out = await agent_r.ainvoke({"messages": [{"role": "user", "content": "What's my savings balance?"}]}, cfg_r)
    reqs = list(fake.requests)   # coordinator, specialist, specialist, coordinator
    check("gpt-oss's reasoning is kept on each reply (vLLM's `reasoning` field), not thrown away",
          out["messages"][1].additional_kwargs.get("reasoning") == "Need the savings balance: ask the accounts agent.")
    check("...and sent back with the turn's earlier tool calls, so the model sees why it made them",
          assistants(reqs[3]["body"])[0].get("reasoning") == "Need the savings balance: ask the accounts agent."
          and assistants(reqs[2]["body"])[0].get("reasoning") == "Call get_balance for savings.")
    fake.script = [{"reasoning": "Polite reply.", "content": "You're welcome."}]
    await agent_r.ainvoke({"messages": [{"role": "user", "content": "Thanks!"}]}, cfg_r)
    check("...but not with past turns (before their final answer), which gpt-oss's format drops anyway",
          assistants(fake.requests[-1]["body"]) and not any(m.get("reasoning") for m in assistants(fake.requests[-1]["body"])))
    fake.script = [{"reasoning": "Thinking while streaming.", "content": "Streamed."}]
    merged = None
    async for piece in model.astream("stream with thinking"):
        merged = piece if merged is None else merged + piece
    check("streamed replies keep the reasoning too",
          merged.additional_kwargs.get("reasoning") == "Thinking while streaming." and merged.content == "Streamed.")
    os.environ["VLLM_SEND_REASONING"] = "off"
    fake.requests.clear()
    fake.script = savings_turn(thinking=True)
    await agent_r.ainvoke({"messages": [{"role": "user", "content": "What's my savings balance?"}]},
                          {"configurable": {"thread_id": "thread-778"}})
    os.environ.pop("VLLM_SEND_REASONING")
    check("VLLM_SEND_REASONING=off sends none back",
          len(fake.requests) == 4 and not any(m.get("reasoning") for r in fake.requests for m in r["body"]["messages"]))

    # ---- a specialist that replies with nothing is asked again (it used to report "" and loop) ----
    from guardrails.middleware import LIMIT_REPLY
    fake.requests.clear()
    fake.script = savings_turn(thinking=False)
    fake.script.insert(1, {"content": ""})                       # the specialist's first reply: empty
    meter = costs.start_turn("thread-779", "A")
    out = await agent_r.ainvoke({"messages": [{"role": "user", "content": "What's my savings balance?"}]},
                                {"configurable": {"thread_id": "thread-779"}, "callbacks": [costs.CostCallback(meter)]})
    costs.end_turn(meter, save=False)
    check("an empty reply (no text, no tool call) is asked again, and the turn still answers",
          len(fake.requests) == 5 and out["messages"][-1].content == f"Your savings balance is £{savings:.2f}."
          and [g["findings"] for g in meter.guard_checks if g["layer"] == "format"] == [["empty_reply"]])

    # ---- the call limit: the customer gets a sentence, not LangChain's "Model call limits exceeded" ----
    fake.script = []
    for _ in range(8):                                           # the coordinator asks the same thing 8 times
        fake.script += [{"tool_calls": [("task", {"subagent_type": "accounts-agent", "description": "Savings balance of the customer."})]},
                        {"content": f"Savings: £{savings:.2f}"}]
    meter = costs.start_turn("thread-780", "A")
    out = await agent_r.ainvoke({"messages": [{"role": "user", "content": "What's my savings balance?"}]},
                                {"configurable": {"thread_id": "thread-780"}, "callbacks": [costs.CostCallback(meter)]})
    costs.end_turn(meter, save=False)
    check("when the call limit stops a turn, the customer gets a plain sentence, not LangChain's message",
          out["messages"][-1].content == LIMIT_REPLY and "exceeded" not in out["messages"][-1].content
          and any(g["layer"] == "limits" for g in meter.guard_checks)
          and any(n.startswith("stopped: too many steps") for n in meter.guard_notes()))

    # ---- another customer gets no reuse of A's cache ----
    meter = costs.start_turn("thread-333", "B")
    fake.script = [{"content": "Hello."}]
    agent_b = await build_coordinator("fake-token-B", InMemorySaver(), bedrock=False)
    await agent_b.ainvoke({"messages": [{"role": "user", "content": "What's my savings balance?"}]},
                          {"configurable": {"thread_id": "thread-333"}, "callbacks": [costs.CostCallback(meter)]})
    costs.end_turn(meter, save=False)
    check("a different customer's identical question reuses none of customer A's cache",
          meter.llm_calls[0]["cache_read"] == 0)

    # ---- the eval report shows the cache share ----
    import contextlib
    import io

    from evals.run import report
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report([{"case_id": "x", "category": "balances", "scores": {"no_error": 1.0}, "comments": {},
                 "outputs": {"cost_usd": 0.0, "latency_ms": 900, "prompt_tokens": 1000, "cached_prompt_tokens": 750,
                             "guard": [{"layer": "format", "mode": "retry"}, {"layer": "format", "mode": "retry"},
                                       {"layer": "format", "mode": "fallback"}]}}],
               {"min_pass_rate": 0.9, "max_avg_cost_usd": 0.01, "max_p95_seconds": 45})
    check("the eval report says how much of the prompt the cache reused",
          "Prefix cache: 75% of prompt tokens reused" in buf.getvalue())
    check("...and how many broken model replies were caught",
          "Broken model replies: 2 caught and asked again, 1 times it still failed" in buf.getvalue())

    print("\nALL vLLM CHECKS PASSED" if OK else "\nSOME CHECKS FAILED")
    return 0 if OK else 1


if __name__ == "__main__":
    servers = subprocess.Popen([sys.executable, "-m", "mcp_servers.run"], cwd=ROOT, env=ENV,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fake = FakeVLLM(port=PORT).start()
    try:
        import socket

        def answering(port: int) -> bool:
            with socket.socket() as sock:
                return sock.connect_ex(("127.0.0.1", port)) == 0

        ports, deadline = [8601, 8602, 8603, 8604, 8605], time.time() + 90
        while time.time() < deadline and servers.poll() is None:
            if all(answering(p) for p in ports):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("the bank servers didn't start (ports 8601-8605)")
        code = asyncio.run(main(fake))
    finally:
        servers.terminate()
        fake.stop()
    sys.exit(code)
