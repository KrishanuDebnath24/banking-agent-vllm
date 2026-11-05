"""E2: how much vLLM's prefix cache saves in a real-looking banking chat.

Sends a scripted 10-message conversation, each message carrying the whole history (as the agents
do), and records for every message the time to the first token (TTFT) and how many prompt tokens
vLLM took from its cache. Run it twice: once as vLLM normally runs, once with prefix caching off.

    python -m bench.cache_on_off --label cache-on
    (restart vLLM with VLLM_PREFIX_CACHING_FLAG=--no-enable-prefix-caching)
    python -m bench.cache_on_off --label cache-off
    python -m bench.summarise

With the cache, TTFT should stay roughly flat as the conversation grows; without it, it rises.

    --base-url    vLLM's OpenAI address (default VLLM_BASE_URL, else http://127.0.0.1:8001/v1)
    --model       served model (default: MODEL's vllm: name, else openai/gpt-oss-20b)
    --session-id  also send x-session-id, so a router keeps the chat on one pod (Kubernetes)
    --salt        send a fresh cache_salt, as the app does per customer: this run shares no cache
                  with any earlier run, so runs compared in E6 (router vs round robin) start equal
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import date
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("BENCH_RESULTS", ROOT / "bench" / "results"))  # /tmp inside the app image

QUESTIONS = [
    "What's my current account balance?",
    "And my savings?",
    "How much did I spend on groceries last month?",
    "Which shop did I spend most at?",
    "Show my last five transactions.",
    "Did I pay rent this month?",
    "How much went on eating out in September?",
    "Was that more than August?",
    "What subscriptions do I pay for?",
    "Thanks. Anything unusual in my spending lately?",
]


def system_prompt() -> str:
    """The real agents' fixed instructions: the same long, unchanging start every call has."""
    from agents import coordinator as c
    from obp import categories

    today = date.today().isoformat()
    parts = [c.COORDINATOR_PROMPT.format(today=today)] + [
        p.format(today=today, categories=categories.for_prompt())
        for p in (c.ACCOUNTS_PROMPT, c.TRANSACTIONS_PROMPT, c.SERVICE_PROMPT, c.CHART_PROMPT)]
    return "\n\n".join(parts)


def default_model() -> str:
    spec = os.environ.get("MODEL", "")
    return spec.split(":", 1)[1] if spec.startswith("vllm:") else "openai/gpt-oss-20b"


def ask(client: OpenAI, model: str, messages: list[dict], headers: dict, max_tokens: int,
        salt: str | None = None) -> dict:
    start = time.perf_counter()
    first, text, usage = None, "", None
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, max_tokens=max_tokens, temperature=0,
        stream_options={"include_usage": True}, extra_headers=headers,
        **({"extra_body": {"cache_salt": salt}} if salt else {}))
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        extra = getattr(delta, "model_extra", None) or {}
        piece = (delta.content or "") or extra.get("reasoning_content") or extra.get("reasoning") or ""
        if piece and first is None:
            first = time.perf_counter() - start
        text += delta.content or ""
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    return {"ttft_ms": round((first or (time.perf_counter() - start)) * 1000, 1),
            "total_ms": round((time.perf_counter() - start) * 1000, 1),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),  # incl. reasoning
            "cached_tokens": getattr(details, "cached_tokens", None) if details else None,
            "answer": text}


def main() -> int:
    from scripts._common import load_env
    if os.environ.get("BANK_TEST") != "1":  # tests pass their own settings and must not pick up a real .env
        load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="cache-on")
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    ap.add_argument("--model", default=default_model())
    ap.add_argument("--session-id", action="store_true", help="send x-session-id (for a router)")
    ap.add_argument("--salt", action="store_true", help="a fresh cache_salt for this run (like a new customer)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("VLLM_API_KEY") or "not-needed")
    headers = {"x-session-id": f"bench-{uuid.uuid4().hex[:8]}"} if args.session_id else {}
    salt = uuid.uuid4().hex if args.salt else None
    if not args.no_warmup:  # load and compile before measuring
        ask(client, args.model, [{"role": "user", "content": "Hello"}], {}, 5)

    # A fresh conversation id in the first message, so an earlier run's cache can't help this one.
    history = [{"role": "system", "content": system_prompt()}]
    rows = []
    print(f"{'#':>2}  {'TTFT ms':>8}  {'prompt':>7}  {'cached':>7}  question")
    for i, q in enumerate(QUESTIONS, 1):
        content = f"[conversation {uuid.uuid4().hex[:8]}] {q}" if i == 1 else q
        history.append({"role": "user", "content": content})
        r = ask(client, args.model, history, headers, args.max_tokens, salt)
        history.append({"role": "assistant", "content": r["answer"] or "OK."})
        rows.append({"turn": i, "question": q, **{k: v for k, v in r.items() if k != "answer"}})
        print(f"{i:>2}  {r['ttft_ms']:>8.0f}  {r['prompt_tokens'] or '-':>7}  {r['cached_tokens'] or 0:>7}  {q}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"e2-{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "model": args.model, "base_url": args.base_url,
                               "turns": rows}, indent=2))
    cached = sum(r["cached_tokens"] or 0 for r in rows)
    prompt = sum(r["prompt_tokens"] or 0 for r in rows)
    if prompt:
        print(f"\nPrompt tokens from the cache: {cached / prompt:.0%} ({cached:,} of {prompt:,})")
    print(f"Saved {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
