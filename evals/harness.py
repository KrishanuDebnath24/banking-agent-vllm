"""Runs one eval case against the real coordinator, subagents and MCP servers.

The servers run on the fixed eval bank (OBP_FAKE + evals/fixture.py) on their own ports, so
they don't clash with the servers you chat with. Each case gets a fresh conversation;
write actions are captured when the agent proposes them and then declined, so the bank's
data is identical for every case.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT_OFFSET = int(os.environ.get("EVAL_PORT_OFFSET", "300"))


def configure_env() -> Path:
    """Point this process at the eval servers. Call before importing agents.coordinator."""
    from evals import fixture

    work = Path(tempfile.mkdtemp(prefix="bank-evals-"))
    fixture_file, demo_file = fixture.write(work)
    os.environ.update({
        "MCP_ACCOUNTS_URL": f"http://127.0.0.1:{8101 + PORT_OFFSET}/mcp",
        "MCP_TRANSACTIONS_URL": f"http://127.0.0.1:{8102 + PORT_OFFSET}/mcp",
        "MCP_SERVICE_URL": f"http://127.0.0.1:{8103 + PORT_OFFSET}/mcp",
        "MCP_CHARTS_URL": f"http://127.0.0.1:{8105 + PORT_OFFSET}/mcp",
        "CHART_DB": str(work / "charts.db"),   # figures saved for charts: the servers write, the API tidies
        "TELEMETRY_DB": "off",
        "EVAL_FIXTURE": str(fixture_file), "EVAL_DEMO": str(demo_file), "EVAL_WORK": str(work)})
    return work


def start_servers() -> subprocess.Popen:
    """Start the eval servers and wait until they accept connections. On failure, the error
    message includes the end of their log, so the reason is visible without opening a temp file."""
    work = Path(os.environ["EVAL_WORK"])
    ports = [8101 + PORT_OFFSET, 8102 + PORT_OFFSET, 8103 + PORT_OFFSET, 8104 + PORT_OFFSET, 8105 + PORT_OFFSET]
    busy = [p for p in ports if _open(p)]
    if busy:
        raise RuntimeError(
            f"ports {busy} are already in use, probably by eval servers left over from an earlier run. "
            "Close them (Task Manager > python.exe, or in PowerShell: "
            f"Get-NetTCPConnection -LocalPort {busy[0]} | % {{ Stop-Process -Id $_.OwningProcess }}) "
            "or set EVAL_PORT_OFFSET=350 in .env.")
    env = {**os.environ, "BANK_TEST": "1", "OBP_FAKE": "1", "OBP_FAKE_FIXTURE": os.environ["EVAL_FIXTURE"],
           "DEMO_ACCOUNTS_FILE": os.environ["EVAL_DEMO"], "MCP_PORT_OFFSET": str(PORT_OFFSET),
           "SERVICE_REQUESTS_DB": str(work / "requests.db"), "OTP_DB": str(work / "otp.db"),
           "SERVICE_REQUESTS_FILE": str(work / "none.jsonl"),  # don't copy in your own old requests
           # The eval bank's records stay in this run's own SQLite files, never in your Postgres.
           "DATABASE_URL": os.environ.get("EVAL_DATABASE_URL", ""),
           "EMAIL_BACKEND": "console", "LANGSMITH_TRACING": "false",
           "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "EVAL_SERVER_PROCESS": "1"}  # log lines appear at once, £ never breaks
    log_path = work / "servers.log"
    log = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - closed when the process ends
    proc = subprocess.Popen([sys.executable, "-m", "mcp_servers.run"], cwd=ROOT, env=env, stdout=log, stderr=log)
    timeout = float(os.environ.get("EVAL_SERVER_TIMEOUT", "120"))  # first start on Windows can be slow
    started = time.time()
    print("Starting eval servers ...", end=" ", flush=True)
    while time.time() - started < timeout:
        if proc.poll() is not None:
            print("failed")
            raise RuntimeError(f"eval servers exited (code {proc.returncode}). End of {log_path}:\n{_tail(log_path)}")
        if all(_open(p) for p in ports):
            print(f"ready in {time.time() - started:.0f}s")
            return proc
        time.sleep(0.5)
    print("timed out")
    stop_servers(proc)
    raise RuntimeError(f"eval servers did not start in {timeout:.0f}s (EVAL_SERVER_TIMEOUT). "
                       f"End of {log_path}:\n{_tail(log_path)}")


def stop_servers(proc: subprocess.Popen) -> None:
    """Stop the servers and make sure the process is really gone, so ports are free next time."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def _tail(path: Path, lines: int = 25) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "(no log)"
    return "\n".join(text.splitlines()[-lines:]) or "(the log is empty: the servers printed nothing yet)"


def _open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content or [])


class Runner:
    """Builds one coordinator per customer and runs cases against it."""

    def __init__(self, model=None):
        from langgraph.checkpoint.memory import InMemorySaver
        self.model, self.saver, self._agents = model, InMemorySaver(), {}
        self._lock = asyncio.Lock()

    async def agent(self, customer: str):
        from agents.coordinator import build_coordinator
        async with self._lock:
            if customer not in self._agents:
                self._agents[customer] = await build_coordinator(f"fake-token-{customer}", self.saver, model=self.model)
            return self._agents[customer]

    async def run(self, inputs: dict) -> dict:
        """inputs: {"case_id", "customer", "turns", "decision"} -> what the evaluators need."""
        from observability import costs
        agent = await self.agent(inputs["customer"])
        thread = f"eval-{inputs['case_id']}-{uuid.uuid4().hex[:6]}"
        config = {"configurable": {"thread_id": thread}, "recursion_limit": 60,
                  "tags": ["eval", inputs["case_id"]], "metadata": {"case_id": inputs["case_id"]}}
        meter = costs.start_turn(thread, inputs["customer"])
        config["callbacks"] = [costs.CostCallback(meter)]
        out = {"answer": "", "subagents": [], "tools": [], "actions": [], "tool_results": [], "error": None}
        try:
            for i, turn in enumerate(inputs["turns"]):
                last = i == len(inputs["turns"]) - 1
                if last:  # score the final turn's behaviour; keep earlier tool results as facts for the judge
                    out.update(earlier_subagents=[s["name"] for s in out["subagents"]], subagents=[], tools=[],
                               actions=[])
                await self._stream(agent, {"messages": [{"role": "user", "content": turn}]}, config, out)
                for _ in range(3):  # answer any confirmations the agent asks for
                    state = await agent.aget_state(config)
                    if not state.interrupts:
                        break
                    resume = {}
                    for intr in state.interrupts:
                        acts = intr.value.get("action_requests", [])
                        out["actions"] += [{"name": a.get("name"), "args": a.get("args", {})} for a in acts]
                        decision = ({"type": "approve"} if inputs.get("decision") == "approve"
                                    else {"type": "reject", "message": "The customer declined this action."})
                        resume[intr.id] = {"decisions": [decision for _ in acts]}
                    from langgraph.types import Command
                    payload = Command(resume=next(iter(resume.values())) if len(resume) == 1 else resume)
                    await self._stream(agent, payload, config, out)
            state = await agent.aget_state(config)
            out["answer"] = _text(state.values["messages"][-1])
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        costs.end_turn(meter, save=False)
        out.update(cost_usd=meter.total_cost, llm_cost_usd=meter.llm_cost, guard_cost_usd=meter.guard_cost,
                   tokens=meter.tokens, model_calls=len(meter.llm_calls), latency_ms=meter.latency_ms,
                   # vLLM's prefix cache: prompt tokens it reused instead of computing again
                   prompt_tokens=sum(c["input_tokens"] for c in meter.llm_calls),
                   cached_prompt_tokens=sum(c["cache_read"] for c in meter.llm_calls),
                   guard=[{k: g[k] for k in ("layer", "source", "mode", "findings", "applied")} for g in meter.guard_checks],
                   unpriced=meter.unpriced)
        return out

    @staticmethod
    def _key(call: dict) -> tuple:
        """A tool call's identity. The id alone isn't enough: ids from a scripted model, or a model
        that reuses them, can clash, and with subgraphs the same message can be streamed twice."""
        return call.get("id"), call.get("name"), json.dumps(call.get("args", {}), sort_keys=True, default=str)

    async def _stream(self, agent, payload, config, out: dict) -> None:
        seen = {self._key({"id": c.get("id"), "name": c["name"], "args": c["args"]}) for c in out["tools"]}
        seen |= {self._key({"id": s.get("id"), "name": "task", "args": s["args"]}) for s in out["subagents"]}
        async for _ns, chunk in agent.astream(payload, config, stream_mode="updates", subgraphs=True):
            for node, data in chunk.items():
                if node == "__interrupt__" or not isinstance(data, dict):
                    continue
                msgs = data.get("messages", [])
                for m in msgs if isinstance(msgs, list) else []:
                    for call in getattr(m, "tool_calls", None) or []:
                        if self._key(call) in seen:
                            continue
                        seen.add(self._key(call))
                        if call["name"] == "task":
                            out["subagents"].append({"id": call.get("id"), "name": call["args"].get("subagent_type"),
                                                     "description": call["args"].get("description", ""),
                                                     "args": call["args"]})
                        else:
                            out["tools"].append({"id": call.get("id"), "name": call["name"], "args": call["args"]})
                    if getattr(m, "type", "") == "tool" and getattr(m, "name", "") != "task":
                        # Kept whole up to 20,000 characters: grounded_amounts looks for every figure in here.
                        out["tool_results"].append(f"{m.name}: {_text(m)[:20000]}")
