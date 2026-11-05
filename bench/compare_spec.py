"""Before/after tables for one vLLM change, such as speculative decoding.

Two kinds of results, compared side by side:

  Sweeps (bench/sweep.sh, on the GPU machine): files <label>-c<N>.json
      python3 -m bench.compare_spec --sweep e1 e1-spec
      python3 -m bench.compare_spec --sweep e3 e3-spec

  Conversations (bench/cache_on_off.py, on your PC): files e2-<label>.json
      python -m bench.compare_spec --chat cache-on spec

For a chat, "writing" is the time from the first token to the last (total minus TTFT): the part
speculative decoding can speed up. With temperature 0 both runs write the same answers, so the
times compare directly. Per-token figures need the token count, which runs from before this
script didn't save; the new run's count is then used for both (same answers).

    --dir   where the result files are (default bench/results)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def _num(d: dict, *keys: str):
    for k in keys:
        if isinstance(d.get(k), (int, float)):
            return float(d[k])
    return None


def _f(v, digits=0) -> str:
    return "-" if v is None else f"{v:,.{digits}f}"


def _ratio(before, after, higher_is_better: bool) -> str:
    """How many times better the new run is (above 1 = better)."""
    if not before or not after:
        return "-"
    return f"{(after / before if higher_is_better else before / after):.2f}x"


def load_sweep(folder: Path, label: str) -> dict[int, dict]:
    runs = {}
    for f in folder.glob(f"{label}-c*.json"):
        m = re.fullmatch(rf"{re.escape(label)}-c(\d+)\.json", f.name)
        if not m:  # "e3-spec-c8.json" must not count as an "e3" result
            continue
        d = json.loads(f.read_text())
        runs[int(m.group(1))] = {
            "tpot_p50": _num(d, "median_tpot_ms", "p50_tpot_ms"),
            "tpot_p95": _num(d, "p95_tpot_ms"),
            "ttft_p50": _num(d, "median_ttft_ms", "p50_ttft_ms"),
            "out_tok_s": _num(d, "output_throughput"),
            "req_s": _num(d, "request_throughput"),
        }
    return runs


def compare_sweeps(folder: Path, before: str, after: str) -> int:
    a, b = load_sweep(folder, before), load_sweep(folder, after)
    if not a or not b:
        missing = [l for l, r in ((before, a), (after, b)) if not r]
        print(f"No sweep results for {', '.join(missing)} in {folder} (files <label>-c<N>.json).")
        return 1
    print(f"Sweep: {before} (before) vs {after} (after). 'Gain' above 1x = the new run is better.\n")
    cols = ["at once", "ms/token p50 before", "after", "gain", "ms/token p95 before", "after",
            "first token p50 ms before", "after", "output tok/s before", "after", "gain"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for c in sorted(set(a) & set(b)):
        x, y = a[c], b[c]
        print(f"| {c} | {_f(x['tpot_p50'], 1)} | {_f(y['tpot_p50'], 1)} | {_ratio(x['tpot_p50'], y['tpot_p50'], False)} "
              f"| {_f(x['tpot_p95'], 1)} | {_f(y['tpot_p95'], 1)} | {_f(x['ttft_p50'])} | {_f(y['ttft_p50'])} "
              f"| {_f(x['out_tok_s'])} | {_f(y['out_tok_s'])} | {_ratio(x['out_tok_s'], y['out_tok_s'], True)} |")
    only = sorted(set(a) ^ set(b))
    if only:
        print(f"\n(Levels in only one run, not shown: {', '.join(map(str, only))})")
    return 0


def load_chat(folder: Path, label: str) -> list[dict] | None:
    f = folder / f"e2-{label}.json"
    return json.loads(f.read_text())["turns"] if f.exists() else None


def compare_chats(folder: Path, before: str, after: str) -> int:
    a, b = load_chat(folder, before), load_chat(folder, after)
    if a is None or b is None:
        missing = [l for l, r in ((before, a), (after, b)) if r is None]
        print(f"No chat results for {', '.join(missing)} in {folder} (files e2-<label>.json).")
        return 1
    print(f"Chat: {before} (before) vs {after} (after). Writing = total minus first token.\n")
    print("| message | tokens | writing ms before | after | ms/token before | after | gain |")
    print("|---|---|---|---|---|---|---|")
    sums = {"wb": 0.0, "wa": 0.0, "tok": 0}
    for i, (x, y) in enumerate(zip(a, b), 1):
        wb = max(0.0, (x.get("total_ms") or 0) - (x.get("ttft_ms") or 0))
        wa = max(0.0, (y.get("total_ms") or 0) - (y.get("ttft_ms") or 0))
        tok = y.get("completion_tokens") or x.get("completion_tokens")
        per_b = wb / (tok - 1) if tok and tok > 1 else None
        per_a = wa / (tok - 1) if tok and tok > 1 else None
        sums["wb"] += wb; sums["wa"] += wa; sums["tok"] += max(0, (tok or 1) - 1)
        print(f"| {i} | {tok or '-'} | {_f(wb)} | {_f(wa)} | {_f(per_b, 1)} | {_f(per_a, 1)} | {_ratio(wb, wa, False)} |")
    if sums["tok"]:
        pb, pa = sums["wb"] / sums["tok"], sums["wa"] / sums["tok"]
        print(f"\nAverage per token: {pb:.1f} ms before, {pa:.1f} ms after "
              f"({1000 / pb:.0f} -> {1000 / pa:.0f} tokens/s), {pb / pa:.2f}x")
    print(f"Writing time, all messages: {sums['wb'] / 1000:.1f} s before, {sums['wa'] / 1000:.1f} s after "
          f"({_ratio(sums['wb'], sums['wa'], False)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sweep", nargs=2, metavar=("BEFORE", "AFTER"), help="two sweep labels, e.g. e3 e3-spec")
    g.add_argument("--chat", nargs=2, metavar=("BEFORE", "AFTER"), help="two E2 labels, e.g. cache-on spec")
    ap.add_argument("--dir", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    if args.sweep:
        return compare_sweeps(args.dir, *args.sweep)
    return compare_chats(args.dir, *args.chat)


if __name__ == "__main__":
    sys.exit(main())
