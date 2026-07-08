#!/usr/bin/env python3
"""Phase 0 benchmark: measure LLM tokens/sec + peak RAM and embedding speed on this machine.

Talks to the local Ollama server over HTTP using only the standard library, so it runs
regardless of the Python venv state. Ollama must be installed and running
(`ollama serve`, default http://localhost:11434) with the candidate models pulled.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"

# Candidate chat models to compare (Phase 0 picks the default from these numbers).
LLM_MODELS = ["llama3.2:3b", "qwen2.5:7b"]
EMBED_MODEL = "nomic-embed-text"

# A fixed, representative prompt so runs are comparable.
PROMPT = (
    "Explain in one short paragraph what retrieval-augmented generation (RAG) is "
    "and why it helps a chatbot answer questions about private documents."
)
# Cap generated tokens so the benchmark is quick but still measures steady-state speed.
NUM_PREDICT = 200

# ~50 short chunks to time the embedder (mimics ingesting a handful of pages).
SAMPLE_CHUNKS = [
    f"Sample chunk number {i}: photosynthesis converts light energy into chemical "
    f"energy stored in glucose, releasing oxygen as a by-product in plant cells."
    for i in range(50)
]


def _post(path: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        f"{OLLAMA}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get(path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(f"{OLLAMA}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _loaded_size_gb(model: str) -> float | None:
    """RAM footprint of a currently-loaded model, per Ollama's /api/ps."""
    try:
        ps = _get("/api/ps")
    except Exception:
        return None
    for m in ps.get("models", []):
        if m.get("name", "").startswith(model) or m.get("model", "").startswith(model):
            return m.get("size", 0) / 1e9
    return None


def _unload(model: str) -> None:
    """Free the model from RAM so the next candidate starts from a clean slate."""
    try:
        _post("/api/generate", {"model": model, "keep_alive": 0}, timeout=60.0)
    except Exception:
        pass


def bench_llm(model: str) -> dict:
    # Warm up (loads the model into RAM; first call includes load time we don't want to
    # count against tokens/sec).
    _post("/api/generate", {"model": model, "prompt": "hi", "stream": False,
                            "options": {"num_predict": 1}})

    wall_start = time.perf_counter()
    resp = _post("/api/generate", {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "options": {"num_predict": NUM_PREDICT},
    })
    wall = time.perf_counter() - wall_start

    eval_count = resp.get("eval_count", 0)
    eval_ns = resp.get("eval_duration", 0) or 1
    prompt_eval_ns = resp.get("prompt_eval_duration", 0) or 0
    tok_per_sec = eval_count / (eval_ns / 1e9)
    size_gb = _loaded_size_gb(model)
    _unload(model)

    return {
        "model": model,
        "tok_per_sec": tok_per_sec,
        "eval_count": eval_count,
        "ttft_s": prompt_eval_ns / 1e9,  # time to process the prompt (proxy for first-token)
        "wall_s": wall,
        "ram_gb": size_gb,
    }


def bench_embeddings(model: str, chunks: list[str]) -> dict:
    # Warm up.
    _post("/api/embeddings", {"model": model, "prompt": chunks[0]})
    start = time.perf_counter()
    for c in chunks:
        _post("/api/embeddings", {"model": model, "prompt": c})
    elapsed = time.perf_counter() - start
    size_gb = _loaded_size_gb(model)
    _unload(model)
    return {
        "model": model,
        "count": len(chunks),
        "total_s": elapsed,
        "chunks_per_sec": len(chunks) / elapsed,
        "ram_gb": size_gb,
    }


def main() -> None:
    # Fail early with a clear message if Ollama isn't reachable.
    try:
        ver = _get("/api/version")
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(f"Cannot reach Ollama at {OLLAMA}: {e}\n"
                         f"Start it with `ollama serve` and pull the models first.")
    print(f"Ollama version: {ver.get('version')}\n")

    print("=== LLM generation ===")
    llm_rows = []
    for m in LLM_MODELS:
        print(f"  benchmarking {m} ...", flush=True)
        try:
            llm_rows.append(bench_llm(m))
        except Exception as e:
            print(f"    ! {m} failed: {e}")

    print(f"\n{'model':<16}{'tok/s':>10}{'gen_tok':>10}{'prompt_s':>10}{'RAM(GB)':>10}")
    print("-" * 56)
    for r in llm_rows:
        ram = f"{r['ram_gb']:.2f}" if r["ram_gb"] is not None else "n/a"
        print(f"{r['model']:<16}{r['tok_per_sec']:>10.1f}{r['eval_count']:>10}"
              f"{r['ttft_s']:>10.2f}{ram:>10}")

    print("\n=== Embeddings (nomic-embed-text) ===")
    emb = bench_embeddings(EMBED_MODEL, SAMPLE_CHUNKS)
    ram = f"{emb['ram_gb']:.2f}" if emb["ram_gb"] is not None else "n/a"
    print(f"  {emb['count']} chunks in {emb['total_s']:.2f}s  "
          f"({emb['chunks_per_sec']:.1f} chunks/s, RAM {ram} GB)")


if __name__ == "__main__":
    main()
