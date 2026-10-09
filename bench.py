#!/usr/bin/env python3
"""Minimal vLLM throughput benchmark.

Fires a fixed prompt at a vLLM server at varying concurrency levels and
reports:
  * per-instance throughput  (tokens/s as seen by each concurrent request)
  * total throughput         (tokens/s produced by the server overall)

Usage example:
  python3 bench.py --parallelism 1 4 8 16 --reps 3 --max-tokens 128
"""

import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import time

import requests

DEFAULT_HOST = "10.14.1.40"
DEFAULT_PORT = 8000
DEFAULT_PROMPT = (
    "Explain, in five short paragraphs, how a transformer attention layer "
    "computes query, key and value projections and what the softmax step does."
)


class ReqResult:
    __slots__ = ("latency", "completion_tokens", "prompt_tokens", "ok", "error")

    def __init__(self):
        self.latency = 0.0
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.ok = False
        self.error = None

    def as_dict(self):
        return {
            "ok": self.ok,
            "latency": round(self.latency, 3),
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "tokens_per_s": round(self.completion_tokens / self.latency, 2)
            if self.ok and self.latency > 0
            else None,
            "error": self.error,
        }


def fire_one(base_url, model, prompt, max_tokens, timeout, min_p=None):
    """Send one chat-completion request; return a ReqResult with timing + usage."""
    result = ReqResult()
    t0 = time.perf_counter()
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        if min_p is not None:
            payload["min_p"] = min_p
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        result.latency = time.perf_counter() - t0
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        result.completion_tokens = int(usage.get("completion_tokens", 0))
        result.prompt_tokens = int(usage.get("prompt_tokens", 0))
        if result.completion_tokens == 0:
            # Fallback: rough estimate if the server omitted usage.
            text = data["choices"][0]["message"]["content"]
            result.completion_tokens = max(1, len(text) // 4)
        result.ok = True
    except Exception as e:  # noqa: BLE001 - report any failure per request
        result.latency = time.perf_counter() - t0
        result.error = repr(e)
    return result


def resolve_model(base_url, model_arg, timeout):
    if model_arg:
        return model_arg
    r = requests.get(f"{base_url}/v1/models", timeout=timeout)
    r.raise_for_status()
    models = r.json().get("data") or []
    if not models:
        raise SystemExit("Server returned no models; pass --model explicitly.")
    return models[0]["id"]


def percentile(sorted_vals, p):
    """Linear-interpolated percentile over a sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def run_level(base_url, model, prompt, max_tokens, parallelism, reps, timeout, min_p=None):
    """Run `reps` rounds of `parallelism` concurrent identical requests."""
    all_results = []
    round_walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=parallelism) as ex:
            futs = [
                ex.submit(fire_one, base_url, model, prompt, max_tokens, timeout, min_p)
                for _ in range(parallelism)
            ]
            all_results.extend(f.result() for f in futs)
        round_walls.append(time.perf_counter() - t0)

    ok = [r for r in all_results if r.ok]
    failed = [r for r in all_results if not r.ok]

    total_tokens = sum(r.completion_tokens for r in ok)
    total_wall = sum(round_walls)
    total_latency = sum(r.latency for r in ok)
    per_rates = [r.completion_tokens / r.latency for r in ok if r.latency > 0]
    latencies = sorted(r.latency for r in ok)

    return {
        "parallelism": parallelism,
        "requests": len(all_results),
        "failed": len(failed),
        "prompt_tokens": ok[0].prompt_tokens if ok else 0,
        "output_tokens_total": total_tokens,
        "output_tokens_mean": total_tokens / len(ok) if ok else 0.0,
        "total_wall_s": round(total_wall, 3),
        "total_throughput_tps": total_tokens / total_wall if total_wall > 0 else 0.0,
        "per_instance_tps_mean": statistics.fmean(per_rates) if per_rates else 0.0,
        "per_instance_tps_share": (
            (total_tokens / total_wall) / parallelism if total_wall > 0 else 0.0
        ),
        "latency_s": {
            "mean": statistics.fmean(latencies) if latencies else float("nan"),
            "p50": percentile(latencies, 50),
            "p99": percentile(latencies, 99),
            "max": latencies[-1] if latencies else float("nan"),
        },
        "avg_output_latency_s": total_latency / len(ok) if ok else float("nan"),
        "errors": [r.error for r in failed[:3]],
        "per_request": [r.as_dict() for r in all_results],
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark a vLLM server with a fixed prompt.")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--model", default=None, help="Model name (default: first from /v1/models)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--prompt-file", default=None, help="Read prompt from file instead of --prompt")
    ap.add_argument("--max-tokens", type=int, default=128, help="Max tokens per completion")
    ap.add_argument("--parallelism", nargs="+", type=int,
                    default=[1, 2, 4, 8, 16],
                    help="Concurrency levels (default: 1 2 4 8 16)")
    ap.add_argument("--reps", type=int, default=2,
                    help="Rounds per parallelism level (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--min-p", type=float, default=None, metavar="F",
                    help="min_p sampling param (fraction of the top token's probability, 0-1); "
                         "off by default")
    ap.add_argument("--json", action="store_true", help="Also print full results as JSON")
    args = ap.parse_args()

    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()
    else:
        prompt = args.prompt

    base_url = f"http://{args.host}:{args.port}"
    try:
        model = resolve_model(base_url, args.model, args.timeout)
    except Exception as e:
        raise SystemExit(f"Could not reach vLLM server at {base_url}: {e!r}")

    levels = list(args.parallelism)
    print(f"Server:   {base_url}")
    print(f"Model:    {model}")
    print(f"Prompt:   {prompt[:60]!r}{'...' if len(prompt) > 60 else ''} "
          f"({len(prompt)} chars)")
    print(f"Max tokens per completion: {args.max_tokens}")
    if args.min_p is not None:
        print(f"min_p:    {args.min_p}")
    print(f"Levels:   {levels}, reps: {args.reps}\n")

    results = []
    for lvl in levels:
        res = run_level(base_url, model, prompt, args.max_tokens, lvl, args.reps, args.timeout,
                        min_p=args.min_p)
        results.append(res)
        lat = res["latency_s"]
        print(f"--- parallelism {lvl} ---")
        print(f"  requests ok/failed : {res['requests'] - res['failed']}/{res['failed']}")
        if res["errors"]:
            print(f"  first error        : {res['errors'][0]}")
        print(f"  output tokens (tot)  : {res['output_tokens_total']} "
              f"(mean {res['output_tokens_mean']:.1f}/req)")
        print(f"  TOTAL throughput     : {res['total_throughput_tps']:8.1f} tok/s "
              f"({res['output_tokens_total']} tok / {res['total_wall_s']:.1f}s wall)")
        print(f"  per-instance tok/s   : {res['per_instance_tps_mean']:8.1f} (mean per request)"
              f"  |  {res['per_instance_tps_share']:8.1f} (total / concurrency)")
        print(f"  latency  p50/p99/max : {lat['p50']:.2f} / {lat['p99']:.2f} / {lat['max']:.2f} s")
        print(f"  avg output lat (req) : {res['avg_output_latency_s']:.2f} s\n")

    if args.json:
        print(json.dumps({"server": base_url, "model": model, "levels": results}, indent=2))


if __name__ == "__main__":
    main()
