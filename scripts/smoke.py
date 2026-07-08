#!/usr/bin/env python3
"""Phase 0 offline smoke test: LlamaIndex <-> Ollama end to end.

Indexes a one-paragraph document with local embeddings, asks a question about it, and
prints the answer. Proves the ingest -> embed -> retrieve -> LLM pipeline works with **no
internet** (Ollama serves everything on localhost).

Run it with networking disabled to prove offline operation, e.g.:
    unshare -rn .venv/bin/python scripts/smoke.py     # Ollama stays reachable on localhost
or toggle Wi-Fi/ethernet off and run:
    .venv/bin/python scripts/smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable so `config` resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

import config

PARAGRAPH = (
    "The Ryzen 7 7730U is a rebranded Zen 3 mobile processor. Despite its 7000-series "
    "name it uses Vega 8 integrated graphics, not the RDNA3 780M found in true Zen 4 "
    "chips. Because Vega 8 is not supported by ROCm, this laptop is treated as a "
    "CPU-inference machine for running local language models."
)
QUESTION = "Why is this laptop treated as a CPU-inference machine?"


def main() -> None:
    Settings.llm = Ollama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=config.REQUEST_TIMEOUT,
        context_window=config.CONTEXT_WINDOW,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )

    index = VectorStoreIndex.from_documents([Document(text=PARAGRAPH)])
    query_engine = index.as_query_engine()

    print(f"Q: {QUESTION}")
    response = query_engine.query(QUESTION)
    print(f"A: {response}")


if __name__ == "__main__":
    main()
