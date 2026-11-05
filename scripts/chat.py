"""Chat with the phase 1 banking agent in the terminal.

    python -m scripts.chat              # as customer A
    python -m scripts.chat --customer B

Type 'quit' to exit. With LANGSMITH_TRACING=true and a LANGSMITH_API_KEY in
.env, every turn appears as a trace in LangSmith.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid

from scripts._common import login, setup
from agents.agent import build_agent
from agents.context import build_context
from agents.llm import model_spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", choices=["A", "B"], default="A")
    args = ap.parse_args()

    client, _ = setup()
    if os.environ.get("LANGSMITH_TRACING", "").strip().lower() != "true":
        print("(LangSmith tracing is OFF: add LANGSMITH_TRACING=true to .env)")
    elif not os.environ.get("LANGSMITH_API_KEY"):
        print("(LangSmith tracing is OFF: no LANGSMITH_API_KEY in .env)")
    else:
        print(f"(LangSmith tracing ON -> project '{os.environ.get('LANGSMITH_PROJECT', 'default')}')")

    username, token = login(client, args.customer)
    ctx = build_context(client, args.customer, username, token)
    agent = build_agent()
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "banking_chat_turn",
        "tags": ["phase1", f"customer-{args.customer}"],
        "metadata": {"customer": args.customer, "thread_id": thread_id},  # groups turns into a thread
    }

    print(f"Logged in as {username} (customer {args.customer}). Model: {model_spec()}")
    print("Ask about your accounts. Type 'quit' to exit.\n")
    seen = 0
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        result = agent.invoke({"messages": [{"role": "user", "content": question}]},
                              config=config, context=ctx)
        messages = result["messages"]
        for m in messages[seen:]:
            for call in getattr(m, "tool_calls", None) or []:
                print(f"  [tool] {call['name']}({json.dumps(call['args'])})")
        seen = len(messages)
        final = messages[-1]
        text = final.text if isinstance(getattr(final, "text", None), str) else str(final.content)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # hide reasoning from some local models
        print(f"bank> {text}\n")

    # Traces are sent in the background; make sure the last ones arrive before exiting.
    from langchain_core.tracers.langchain import wait_for_all_tracers
    wait_for_all_tracers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
