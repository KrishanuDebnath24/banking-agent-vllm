"""A stand-in for a vLLM server, for offline tests: the OpenAI chat API with scripted answers.

It records every request (headers and body), answers from a script (text or tool calls, in
order), and simulates vLLM's prefix cache well enough to test against: a request's cached tokens
are the longest start it shares with an earlier request that carried the same cache_salt.
Prompts are measured in "tokens" of 4 characters, cached in blocks of 16, like vLLM's full blocks.

    server = FakeVLLM(port=8399).start()
    server.script = [{"tool_calls": [("get_balance", {"account": "savings"})]}, {"content": "£10.00"}]
    ... point VLLM_BASE_URL at server.url ...
    server.requests   # what the app sent
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

BLOCK = 16


def _render(body: dict) -> str:
    """The prompt as one string that only grows when messages are appended (like a chat template)."""
    head = json.dumps(body.get("tools") or [], sort_keys=True)
    return head + "".join("\n" + json.dumps(m, sort_keys=True) for m in body.get("messages") or [])


def _common(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class FakeVLLM:
    def __init__(self, port: int = 8399, model: str = "test-model"):
        self.port, self.model = port, model
        self.url = f"http://127.0.0.1:{port}/v1"
        self.script: list[dict] = []
        self.requests: list[dict] = []
        self.seen: dict[str, list[str]] = {}       # cache salt -> prompts already "computed"
        self.prefix_caching = True
        self.app = self._build()
        self._server: uvicorn.Server | None = None

    # ---- the simulated cache ----
    def _usage(self, body: dict, completion_tokens: int) -> dict:
        prompt = _render(body)
        salt = str(body.get("cache_salt") or "")
        best = max((_common(prompt, old) for old in self.seen.get(salt, [])), default=0) if self.prefix_caching else 0
        self.seen.setdefault(salt, []).append(prompt)
        tokens = len(prompt) // 4
        cached = min((best // 4) // BLOCK * BLOCK, tokens)
        return {"prompt_tokens": tokens, "completion_tokens": completion_tokens,
                "total_tokens": tokens + completion_tokens, "prompt_tokens_details": {"cached_tokens": cached}}

    def _next(self) -> dict:
        return self.script.pop(0) if self.script else {"content": "OK."}

    def _build(self) -> FastAPI:
        app = FastAPI()

        @app.get("/v1/models")
        async def models():
            return {"object": "list", "data": [{"id": self.model, "object": "model"}]}

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            body = await request.json()
            self.requests.append({"headers": {k.lower(): v for k, v in request.headers.items()}, "body": body})
            reply = self._next()
            if reply.get("delay"):
                await asyncio.sleep(reply["delay"])
            calls = [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args)}}
                     for name, args in reply.get("tool_calls", [])]
            content = reply.get("content", "" if calls else "OK.")
            usage = self._usage(body, completion_tokens=max(1, len(content) // 4 + 10 * len(calls)))
            finish = "tool_calls" if calls else "stop"
            base = {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "created": int(time.time()), "model": self.model}
            reasoning = reply.get("reasoning")      # vLLM's field for gpt-oss's thinking
            if not body.get("stream"):
                message = {"role": "assistant", "content": content or None}
                if reasoning:
                    message["reasoning"] = reasoning
                if calls:
                    message["tool_calls"] = calls
                return JSONResponse({**base, "object": "chat.completion", "usage": usage,
                                     "choices": [{"index": 0, "message": message, "finish_reason": finish}]})

            def chunk(delta: dict, finish_reason=None) -> str:
                return "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}]}) + "\n\n"

            async def events():
                yield chunk({"role": "assistant", "content": ""})
                for i in range(0, len(reasoning or ""), 8):
                    yield chunk({"reasoning": reasoning[i:i + 8]})
                for i in range(0, len(content), 8):
                    await asyncio.sleep(0.005)
                    yield chunk({"content": content[i:i + 8]})
                for i, c in enumerate(calls):
                    yield chunk({"tool_calls": [{"index": i, **c}]})
                yield chunk({}, finish)
                if (body.get("stream_options") or {}).get("include_usage"):
                    yield "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [],
                                                 "usage": usage}) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")

        return app

    # ---- run in a background thread ----
    def start(self) -> "FakeVLLM":
        config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        threading.Thread(target=self._server.run, daemon=True).start()
        deadline = time.time() + 15
        while not self._server.started and time.time() < deadline:
            time.sleep(0.05)
        return self

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
