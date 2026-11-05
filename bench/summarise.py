"""Turn the benchmark results in bench/results/ into tables to paste into the plan's report.

    python -m bench.summarise              # every result found
    python -m bench.summarise --csv        # the same, as CSV

Reads `vllm bench serve` results (bench/sweep.sh) and the E2 conversation runs
(bench/cache_on_off.py). Field names differ a little between vLLM versions, so anything a result
doesn't have shows as "-".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def _num(d: dict, *keys: str):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _fmt(v, digits=0) -> str:
    if v is None:
        return "-"
    return f"{v:,.{digits}f}" if isinstance(v, (int, float)) else str(v)


def sweep_rows(files: list[Path]) -> list[dict]:
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        if "request_throughput" not in d:
            continue
        m = re.search(r"-c(\d+)\.json$", f.name)
        conc = _num(d, "max_concurrency") or (int(m.group(1)) if m else None)
        done, total = d.get("completed"), _num(d, "num_prompts") or d.get("completed")
        rows.append({
            "run": f.stem, "concurrency": conc,
            "req_per_s": d.get("request_throughput"), "output_tok_per_s": d.get("output_throughput"),
            "ttft_p50_ms": _num(d, "p50_ttft_ms", "median_ttft_ms"), "ttft_p95_ms": d.get("p95_ttft_ms"),
            "itl_p95_ms": d.get("p95_itl_ms"), "tpot_p95_ms": d.get("p95_tpot_ms"),
            "e2e_p95_ms": d.get("p95_e2el_ms"), "goodput_req_per_s": d.get("request_goodput"),
            "errors": (total - done) if isinstance(total, int) and isinstance(done, int) else None})
    return sorted(rows, key=lambda r: (re.sub(r"-c\d+$", "", r["run"]), r["concurrency"] or 0))


def conversation_runs(files: list[Path]) -> dict[str, dict]:
    runs = {}
    for f in files:
        d = json.loads(f.read_text())
        if "turns" in d:
            runs[d.get("label", f.stem)] = d
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--dir", type=Path, default=RESULTS)
    ap.add_argument("--from-log", type=Path, help="a Kubernetes Job's log (deploy/k8s/loadtest): its "
                                                  "RESULT_JSON lines are saved to --dir first")
    args = ap.parse_args()
    if args.from_log:
        args.dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for line in args.from_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("RESULT_JSON "):
                _, name, payload = line.split(" ", 2)
                (args.dir / Path(name).name).write_text(payload)
                saved += 1
        print(f"Saved {saved} result(s) from {args.from_log} to {args.dir}\n")
    files = sorted(args.dir.glob("*.json"))
    if not files:
        print(f"No results in {args.dir}. Run bench/sweep.sh or python -m bench.cache_on_off first.")
        return 1

    rows = sweep_rows(files)
    if rows:
        cols = ["run", "concurrency", "req_per_s", "output_tok_per_s", "ttft_p50_ms", "ttft_p95_ms",
                "itl_p95_ms", "e2e_p95_ms", "goodput_req_per_s", "errors"]
        if args.csv:
            print(",".join(cols))
            for r in rows:
                print(",".join("" if r[c] is None else str(r[c]) for c in cols))
        else:
            print("Throughput and latency by concurrency (E1, E3, E4, E5)\n")
            print("| " + " | ".join(cols) + " |")
            print("|" + "---|" * len(cols))
            for r in rows:
                print("| " + " | ".join(_fmt(r[c], 1 if c in ("req_per_s", "goodput_req_per_s") else 0)
                                        if c != "run" else r[c] for c in cols) + " |")
            ok = [r for r in rows if (r["ttft_p95_ms"] or 1e9) <= 1000 and (r["itl_p95_ms"] or 1e9) <= 50
                  and not r["errors"]]
            if ok:
                best = max(ok, key=lambda r: r["concurrency"] or 0)
                print(f"\nHighest concurrency within p95 TTFT 1 s and p95 ITL 50 ms: {best['concurrency']} "
                      f"({best['run']}, {_fmt(best['output_tok_per_s'])} output tokens/s)")

    runs = conversation_runs(files)
    if runs:
        print("\nTime to first token per message, by run (E2)\n")
        labels = sorted(runs)
        print("| message | " + " | ".join(f"{l} TTFT ms | {l} cached" for l in labels) + " |")
        print("|---|" + "---|---|" * len(labels))
        turns = max(len(runs[l]["turns"]) for l in labels)
        for i in range(turns):
            cells = []
            for l in labels:
                t = runs[l]["turns"][i] if i < len(runs[l]["turns"]) else {}
                cells += [_fmt(t.get("ttft_ms")), f"{_fmt(t.get('cached_tokens'))} of {_fmt(t.get('prompt_tokens'))}"]
            print(f"| {i + 1} | " + " | ".join(cells) + " |")
        for l in labels:
            ts = runs[l]["turns"]
            cached = sum(t.get("cached_tokens") or 0 for t in ts)
            prompt = sum(t.get("prompt_tokens") or 0 for t in ts)
            share = f"{cached / prompt:.0%} of prompt tokens from the cache" if prompt else "no token counts"
            print(f"\n{l}: average TTFT {_fmt(sum(t['ttft_ms'] for t in ts) / len(ts))} ms, {share}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
