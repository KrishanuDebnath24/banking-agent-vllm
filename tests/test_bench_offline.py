"""Offline test of the benchmark tools against the fake vLLM server (no GPU needed).

    python tests/test_bench_offline.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.fake_vllm import FakeVLLM  # noqa: E402

OK = True
PORT = 8697


def check(label, cond):
    global OK
    OK &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label)


def run(*args, env=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, timeout=120)


def main() -> int:
    fake = FakeVLLM(port=PORT).start()
    results = Path(tempfile.mkdtemp(prefix="bench-results-"))
    # BANK_TEST=1: the tools skip your real .env (its VLLM_BASE_URL points at the GPU machine, not the fake)
    env = {"BANK_TEST": "1", "VLLM_BASE_URL": fake.url, "MODEL": "vllm:test-model", "LANGSMITH_TRACING": "false"}
    import bench.cache_on_off as e2
    import bench.summarise as summ
    e2.RESULTS, summ.RESULTS = results, results

    try:
        # E2 with the cache on, then off, in-process so the results land in the temp folder
        for label, caching in (("cache-on", True), ("cache-off", False)):
            fake.prefix_caching, fake.seen = caching, {}
            fake.script = [{"content": f"Answer {i}.", "delay": 0.01} for i in range(20)]
            os.environ.update(env)
            sys.argv = ["cache_on_off", "--label", label, "--no-warmup", "--base-url", fake.url, "--model", "test-model"]
            assert e2.main() == 0
        on = json.loads((results / "e2-cache-on.json").read_text())["turns"]
        off = json.loads((results / "e2-cache-off.json").read_text())["turns"]
        check("E2 records all 10 messages, each with TTFT and prompt tokens",
              len(on) == 10 and all(t["ttft_ms"] > 0 and t["prompt_tokens"] for t in on))
        check("E2 sends the agents' real system prompt, the whole history each time, and a growing prompt",
              "digital assistant of a UK retail bank" in fake.requests[-1]["body"]["messages"][0]["content"]
              and len(fake.requests[-1]["body"]["messages"]) == 1 + 2 * 9 + 1
              and on[-1]["prompt_tokens"] > on[0]["prompt_tokens"])
        check("with the cache on, later messages reuse most of the prompt; with it off, none",
              all((t["cached_tokens"] or 0) > 0.8 * t["prompt_tokens"] for t in on[1:])
              and all(not t["cached_tokens"] for t in off))

        # --salt: a fresh cache_salt per run, so a second run can't reuse the first run's cache (E6)
        fake.prefix_caching = True
        for label, extra in (("plain-2", []), ("salted", ["--salt"])):
            fake.script = [{"content": f"Answer {i}.", "delay": 0.01} for i in range(20)]
            sys.argv = ["cache_on_off", "--label", label, "--no-warmup", "--base-url", fake.url, "--model", "test-model", *extra]
            assert e2.main() == 0
        plain = json.loads((results / "e2-plain-2.json").read_text())["turns"]
        salted = json.loads((results / "e2-salted.json").read_text())["turns"]
        check("--salt: a run shares no cache with earlier runs, but still reuses its own conversation",
              (plain[0]["cached_tokens"] or 0) > 0 and not salted[0]["cached_tokens"]
              and all((t["cached_tokens"] or 0) > 0.8 * t["prompt_tokens"] for t in salted[1:])
              and fake.requests[-1]["body"].get("cache_salt"))

        # E3-style result files, with the field names vllm bench serve writes (checked in v0.32.0's source)
        for c, ttft, itl in ((8, 300.0, 30.0), (64, 900.0, 45.0), (128, 2400.0, 80.0)):
            (results / f"e3-c{c}.json").write_text(json.dumps({
                "completed": c * 10, "num_prompts": c * 10, "max_concurrency": c, "request_throughput": c / 4,
                "output_throughput": c * 50.0, "p50_ttft_ms": ttft / 2, "p95_ttft_ms": ttft, "p95_itl_ms": itl,
                "p95_tpot_ms": itl, "p95_e2el_ms": 9000.0, "request_goodput": c / 5}))
        out = run("bench.summarise", "--dir", str(results), env=env)
        check("summarise prints the sweep table and the E2 comparison",
              out.returncode == 0 and "| e3-c64 | 64 |" in out.stdout and "cache-on TTFT ms" in out.stdout)
        check("...and names the highest concurrency that stays within both latency targets",
              "within p95 TTFT 1 s and p95 ITL 50 ms: 64" in out.stdout)
        check("...and the share of prompt tokens the cache served",
              "cache-off: average TTFT" in out.stdout and "of prompt tokens from the cache" in out.stdout)
        csv = run("bench.summarise", "--dir", str(results), "--csv", env=env)
        check("summarise --csv gives one line per run", csv.stdout.count("\ne3-c") == 3)

        # compare_spec: before/after tables (speculative decoding and similar changes)
        check("E2 also records the answer's token count", all(t.get("completion_tokens") for t in on))
        for c, tpot in ((8, 20.0), (64, 40.0)):
            (results / f"e3-c{c}.json").write_text(json.dumps({
                **json.loads((results / f"e3-c{c}.json").read_text()), "median_tpot_ms": tpot}))
            (results / f"e3-spec-c{c}.json").write_text(json.dumps({
                "completed": c * 10, "num_prompts": c * 10, "max_concurrency": c, "request_throughput": c / 3,
                "output_throughput": c * 80.0, "median_ttft_ms": 140.0, "median_tpot_ms": tpot / 2,
                "p95_tpot_ms": tpot}))
        cmp_sweep = run("bench.compare_spec", "--dir", str(results), "--sweep", "e3", "e3-spec", env=env)
        check("compare_spec --sweep lines up the levels in both runs and shows the gain",
              cmp_sweep.returncode == 0 and "| 8 | 20.0 | 10.0 | 2.00x |" in cmp_sweep.stdout
              and "| 64 | 40.0 | 20.0 | 2.00x |" in cmp_sweep.stdout and "not shown: 128" in cmp_sweep.stdout)
        cmp_chat = run("bench.compare_spec", "--dir", str(results), "--chat", "cache-on", "salted", env=env)
        check("compare_spec --chat compares writing time and time per token, message by message",
              cmp_chat.returncode == 0 and "Average per token:" in cmp_chat.stdout
              and cmp_chat.stdout.count("\n| ") >= 10)
        missing = run("bench.compare_spec", "--dir", str(results), "--sweep", "e3", "nope", env=env)
        check("compare_spec says which run is missing", missing.returncode == 1 and "nope" in missing.stdout)
        sweep = (ROOT / "bench" / "sweep.sh").read_text()
        check("sweep.sh uses only flags that vllm bench serve v0.32.0 has",
              all(f in sweep for f in ("--backend openai-chat", "--random-prefix-len", "--ignore-eos",
                                       "--goodput", "--result-dir", "--percentile-metrics ttft,tpot,itl,e2el")))
    finally:
        fake.stop()

    print("\nALL BENCHMARK TOOL CHECKS PASSED" if OK else "\nSOME CHECKS FAILED")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
