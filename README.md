# ai-benchmarker

Minimum viable throughput benchmark for a vLLM server. Fires one fixed
prompt at the server at a series of concurrency levels and reports:

- **Total throughput** — output tokens / wall time for the whole round
  (what the server as a whole is doing).
- **Per-instance throughput** — tokens/s as seen by each concurrent request
  (mean per request, and total ÷ concurrency for comparison).
- Request latency (p50/p99/max), token counts, and per-request detail with `--json`.

Token counts come from the server's `usage.completion_tokens` (accurate),
with a crude `len(text)//4` fallback if the server omits usage.

## Usage

    python3 bench.py --parallelism 1 4 8 16 --reps 3

Defaults target a vLLM server at `10.14.1.40:8000` (auto-detects the model
from `/v1/models`).

Useful flags:

| flag | default | meaning |
|---|---|---|
| `--host` / `--port` | `10.14.1.40` / `8000` | server location |
| `--model` | first from `/v1/models` | model name |
| `--prompt` / `--prompt-file` | a 5-paragraph transformer explainer | the prompt to benchmark |
| `--max-tokens` | `128` | completion budget per request |
| `--min-p` | off | sampling param: drop tokens below this fraction of the top token's probability (0–1) |
| `--parallelism 1 4 8 ...` | `1 2 4 8 16` | concurrency levels to test |
| `--reps` | `2` | rounds per level (more = steadier numbers) |
| `--timeout` | `300` | per-request HTTP timeout (s) |
| `--json` | off | print full per-request results as JSON |

## Example output

    --- parallelism 8 ---
      requests ok/failed : 16/0
      output tokens (tot)  : 2048 (mean 128.0/req)
      TOTAL throughput     :    209.8 tok/s (2048 tok / 9.8s wall)
      per-instance tok/s   :     26.3 (mean per request)  |     26.2 (total / concurrency)
      latency  p50/p99/max : 4.87 / 4.88 / 4.88 s

## Caveats

- All requests use the same prompt/length, so this measures decode-phase
  scaling on uniform load, not mixed workloads.
- `temperature=0` for reproducibility.
- Client-side measurement: request latency includes network RTT, and total
  throughput is measured from the client, so it slightly undercounts what
  the GPU actually produced.
