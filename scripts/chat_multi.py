"""Chat with the phase 2 multi-agent assistant (coordinator + 3 subagents over MCP).

Start the MCP servers first, in another terminal:
    python -m mcp_servers.run

Then:
    python -m scripts.chat_multi                    # customer A, new conversation
    python -m scripts.chat_multi --customer B
    python -m scripts.chat_multi --thread A-1234    # continue an earlier conversation

Transfers and service requests stop and ask you to confirm (y/n) before they run.
Address changes and larger transfers then ask for the security code emailed to you;
the code goes straight to the bank's verify service, never to the AI.
Conversations, including any pending confirmation, are saved in sessions.db
(or in Postgres when DATABASE_URL is set).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import uuid

import httpx
from langgraph.types import Command

from scripts._common import load_env, login, setup
import storage
from agents.coordinator import build_coordinator
from agents.llm import model_spec
from guardrails.bedrock import BedrockGuard
from observability import costs

CODE_NOT_CONFIRMED = "The security code was not confirmed, so nothing was done."


def describe(action: dict) -> str:
    """Plain-English summary of a pending write action."""
    a, name = action.get("args", {}), action.get("name", "")
    try:
        amount = f"£{float(a.get('amount', 0)):.2f}"
    except (TypeError, ValueError):
        amount = str(a.get("amount"))
    return {
        "transfer_between_my_accounts": f"Transfer {amount} from {a.get('from_account')} to {a.get('to_account')}"
                                        + (f" (reference: {a['reference']})" if a.get("reference") else ""),
        "change_address": f"Change your address to: {a.get('new_address')}",
        "request_cheque_book": f"Order a new cheque book for your {a.get('account')} account",
        "request_kyc_update": f"Update your identity details: {a.get('update')}",
    }.get(name, f"{name}({a})")


def text_of(message) -> str:
    text = message.text if isinstance(getattr(message, "text", None), str) else str(message.content)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


async def run(agent, payload, config, shown: set) -> None:
    """Stream one step, printing which agent is doing what (each tool call once)."""
    async for ns, chunk in agent.astream(payload, config, stream_mode="updates", subgraphs=True):
        for node, data in chunk.items():
            if node == "__interrupt__" or not isinstance(data, dict):
                continue
            msgs = data.get("messages", [])
            for m in msgs if isinstance(msgs, list) else []:
                for call in getattr(m, "tool_calls", None) or []:
                    key = (call.get("id"), call.get("name"), repr(call.get("args")))
                    if key in shown:
                        continue
                    shown.add(key)
                    if call["name"] == "task":
                        print(f"  -> {call['args'].get('subagent_type')}: {call['args'].get('description', '')[:110]}")
                    else:
                        args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                        print(f"       [tool] {call['name']}({args[:110]})")


async def security_code(token: str, action: dict) -> bool:
    """If this action needs a security code, have one emailed and check what the customer types.

    The code goes from the keyboard straight to the verify service. It is never put in the
    conversation, so the AI model, the session store and the traces never see it.
    Returns True when the action may go ahead.
    """
    headers = {"Authorization": f'DirectLogin token="{token}"'}
    verify_url = os.environ.get("VERIFY_URL", "http://127.0.0.1:8104")  # read after .env is loaded
    async with httpx.AsyncClient(base_url=verify_url, headers=headers, timeout=30) as http:
        try:
            r = await http.post("/verify/start", json={"tool": action.get("name"), "args": action.get("args", {})})
            data = r.json()
        except (httpx.HTTPError, ValueError):
            print("  Could not reach the security code service. Is  python -m mcp_servers.run  running?")
            return False
        if r.status_code != 200:
            print(f"  Could not send a security code: {data.get('message', r.status_code)}")
            return False
        if not data.get("required"):
            return True
        print(f"  We've emailed a 6-digit security code to {data['sent_to']}. "
              f"It expires in {data['expires_in'] // 60} minutes. (Press Enter to cancel.)")
        while True:
            code = (await asyncio.to_thread(input, "  Security code: ")).strip()
            if not code:
                return False
            try:
                res = (await http.post("/verify/check", json={"challenge_id": data["challenge_id"], "code": code})).json()
            except (httpx.HTTPError, ValueError):
                print("  Could not reach the security code service.")
                return False
            if res.get("verified"):
                return True
            if res.get("attempts_left", 0) <= 0:
                print("  That code has expired or had too many wrong tries.")
                return False
            print(f"  That code isn't right. {res['attempts_left']} tries left.")


async def confirm_pending(agent, config, shown: set, token: str) -> bool:
    """Ask about every pending action. Returns True if something was resumed."""
    state = await agent.aget_state(config)
    if not state.interrupts:
        return False
    answers = {}
    for intr in state.interrupts:
        decisions = []
        for action in intr.value.get("action_requests", []):
            reply = (await asyncio.to_thread(input, f"\n  CONFIRM: {describe(action)}\n  Go ahead? [y/N] ")).strip().lower()
            if reply not in ("y", "yes"):
                decisions.append({"type": "reject", "message": "The customer declined this action."})
            elif await security_code(token, action):
                decisions.append({"type": "approve"})
            else:
                decisions.append({"type": "reject", "message": CODE_NOT_CONFIRMED})
        answers[intr.id] = {"decisions": decisions}
    resume = next(iter(answers.values())) if len(answers) == 1 else answers
    await run(agent, Command(resume=resume), config, shown)
    return True


def with_meter(config: dict, meter) -> dict:
    """The turn's config: the cost meter sees every model call, subagents included."""
    return {**config, "callbacks": [costs.CostCallback(meter)]}


def report(meter) -> None:
    for note in meter.guard_notes():
        print(f"       [guardrail] {note}")
    print(f"       [cost] {meter.summary_line()}\n")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", choices=["A", "B"], default="A")
    ap.add_argument("--thread", help="continue an earlier conversation, e.g. A-1a2b3c4d")
    args = ap.parse_args()

    client, _ = setup()
    username, token = login(client, args.customer)
    thread_id = args.thread or f"{args.customer}-{uuid.uuid4().hex[:8]}"
    if not thread_id.startswith(f"{args.customer}-"):
        print(f"Thread {thread_id} belongs to another customer.")
        return 1

    tracing = os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and os.environ.get("LANGSMITH_API_KEY")
    trace_note = ""
    if tracing:
        from observability.tracing import install_trace_redaction
        trace_note = f" ({install_trace_redaction()})"
    bedrock = BedrockGuard.from_env()
    config = {"configurable": {"thread_id": thread_id}, "run_name": "banking_multi_agent_turn",
              "tags": ["phase3", f"customer-{args.customer}"],
              "metadata": {"customer": args.customer, "thread_id": thread_id,
                           "guardrail_mode": bedrock.mode if bedrock else "rules-only"}}

    async with storage.checkpointer() as saver:
        try:
            agent = await build_coordinator(token, saver)
        except Exception as e:  # noqa: BLE001
            print(f"Could not reach the MCP servers ({type(e).__name__}). Start them first:  python -m mcp_servers.run")
            return 1

        print(f"Logged in as {username} (customer {args.customer}). Model: {model_spec()}")
        print(f"Conversation: {thread_id}   (continue later with --thread {thread_id})")
        print("Guardrails: rules ON; Bedrock " + (f"{bedrock.mode.upper()} ({bedrock.guardrail_id} v{bedrock.version})"
                                                  if bedrock else "OFF (set GUARDRAIL_ID to turn on)"))
        print("LangSmith tracing " + ("ON" + trace_note if tracing else "OFF") + ". Type 'quit' to exit.\n")

        # A conversation saved mid-confirmation picks up where it left off.
        meter = costs.start_turn(thread_id, args.customer)
        if await confirm_pending(agent, with_meter(config, meter), set(), token):
            state = await agent.aget_state(config)
            print(f"bank> {text_of(state.values['messages'][-1])}")
            report(costs.end_turn(meter))

        while True:
            try:
                question = (await asyncio.to_thread(input, "you> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"quit", "exit"}:
                break
            if not question:
                continue
            shown: set = set()
            meter = costs.start_turn(thread_id, args.customer)
            turn = with_meter(config, meter)
            await run(agent, {"messages": [{"role": "user", "content": question}]}, turn, shown)
            while await confirm_pending(agent, turn, shown, token):
                pass
            state = await agent.aget_state(config)
            print(f"bank> {text_of(state.values['messages'][-1])}")
            report(costs.end_turn(meter))

    from langchain_core.tracers.langchain import wait_for_all_tracers
    wait_for_all_tracers()
    return 0


if __name__ == "__main__":
    try:
        load_env()
        code = asyncio.run(main(), loop_factory=storage.loop_factory())
    except KeyboardInterrupt:  # Ctrl+C: leave quietly, the conversation is already saved
        print("\nBye. Your conversation is saved; continue it with --thread.")
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
        code = 0
    raise SystemExit(code)
