# AI Doubt Solver (Local RAG)

Ask "doubts" about your own PDFs and images — a local, **fully offline** retrieval-augmented
generation (RAG) "chat with your documents" tool. Everything (LLM, embeddings, vector store) runs on
this laptop; no data leaves the machine.

See [`plan.md`](./plan.md) for the architecture and phased plan, and [`tasks.md`](./tasks.md) for the
task checklist. This README is a setup stub, fleshed out in Phase 6.

## Requirements

- **Ollama** (local model server) — https://ollama.com
- **Python 3.12** — the ML/OCR deps (onnxruntime, chromadb, PyMuPDF) may lack wheels for very new
  Python releases (e.g. 3.14). We pin 3.12 via [`uv`](https://docs.astral.sh/uv/).

## Setup

```bash
# 1. Install uv (one-time), then fetch Python 3.12 and create the venv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 .venv

# 2. Install Python dependencies
uv pip install -r requirements.txt

# 3. Pull the local models (Ollama must be installed & running)
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

## Phase 0 scripts

```bash
# Benchmark candidate LLMs (tokens/sec + peak RAM) and embedding speed on this machine
.venv/bin/python scripts/benchmark.py

# Offline smoke test: index a paragraph and answer a question with no network
.venv/bin/python scripts/smoke.py
```

Benchmark results and the chosen default model live in [`docs/benchmarks.md`](./docs/benchmarks.md);
settings live in [`config.py`](./config.py).

## Offline note

At runtime nothing needs the internet — Ollama serves models locally at `http://localhost:11434`.
The Phase 0 smoke test proves this by answering with networking disabled.
