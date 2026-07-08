# Phase 0 Benchmarks — 7730U, CPU-only

Measured with `scripts/benchmark.py` (talks to local Ollama over HTTP), on the target machine.

## Machine

- **CPU:** AMD Ryzen 7 7730U (Zen 3, 16 threads), CPU inference only (Vega 8 iGPU, no ROCm).
- **RAM:** 14 GiB total, ~8.4 GiB free at benchmark time.
- **OS / runtime:** Ubuntu 26.04 LTS · Ollama 0.31.1 · Python 3.12.13 (via `uv`).

## Results

Fixed prompt (RAG explainer), `num_predict=200`. `tok/s` = `eval_count / eval_duration` from
Ollama; `prompt_s` = prompt-eval time (first-token proxy); `RAM(GB)` = model footprint from
`/api/ps` while loaded.

| Model         | tok/s | gen tokens | prompt_s | RAM (GB) |
|---------------|------:|-----------:|---------:|---------:|
| **llama3.2:3b** | **16.3** | 82 | 0.33 | **2.56** |
| qwen2.5:7b    |  7.9  | 91 | 0.84 | 5.06 |

**Embeddings** (`nomic-embed-text`): 50 chunks in **2.36 s** → **21.2 chunks/s**, ~0.38 GB RAM.

## Decision — default model: `llama3.2:3b`

- **~2× faster** (16.3 vs 7.9 tok/s) and **half the RAM** (2.56 vs 5.06 GB) than qwen2.5:7b. Both fit
  the ~9 GB budget, but 3B leaves comfortable headroom for Chroma + capped chat memory + other apps.
- Prompt processing is sub-second for both, so first-token latency is dominated by model load (kept
  warm via `keep_alive`), not compute.
- `qwen2.5:7b` remains the **quality fallback** — it also fits RAM; switch `LLM_MODEL` in `config.py`
  if grounded-answer quality proves insufficient on 3B. (Run only one model loaded at a time.)

### Latency target (requirement E4)

With `llama3.2:3b` at ~16 tok/s and sub-second prompt processing:
- **Target:** a typical grounded answer (~150–250 tokens) returns within **~15 s** of retrieval, with
  first output under ~2 s once the model is warm.
- Embedding ingest runs at ~20 chunks/s, so a handful of pages indexes in seconds.

`qwen2.5:7b` roughly doubles answer time (~25–30 s for the same length) — acceptable only if quality
demands it.
