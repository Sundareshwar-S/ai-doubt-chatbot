# AI Doubt Solver (Local RAG)

Ask "doubts" about your own PDFs and images — a local, **fully offline** retrieval-augmented
generation (RAG) "chat with your documents" tool. Everything (LLM, embeddings, vector store) runs on
this laptop; no data leaves the machine. The backend is FastAPI; the UI is a React + Vite app.

See [`plan.md`](./plan.md) for the architecture and phased plan, and [`tasks.md`](./tasks.md) for the
task checklist.

## Requirements

- **Ollama** (local model server) — https://ollama.com
- **Python 3.12** — the ML/OCR deps (onnxruntime, chromadb, PyMuPDF) may lack wheels for very new
  Python releases (e.g. 3.14). We pin 3.12 via [`uv`](https://docs.astral.sh/uv/).
- **Node.js 20+** — for the React/Vite frontend (`frontend/`).

## Setup

```bash
# 1. Install uv (one-time), then fetch Python 3.12 and create the venv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 .venv

# 2. Install Python dependencies (the venv has no pip binary; use `uv pip`)
uv pip install --python .venv/bin/python -r requirements.txt

# 3. Pull the local models (Ollama must be installed & running)
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 4. Install frontend dependencies
cd frontend && npm install && cd ..
```

## Running the app

The backend serves the API at `http://localhost:8000` (`/api/documents`, `/api/chat/*`, `/health`).
There are two ways to run the app; both need Ollama already running (`ollama serve`, or the desktop
app) with the models above pulled.

### Dev mode (two terminals, hot reload)

```bash
# Terminal 1 — backend, auto-reloads on Python changes
.venv/bin/uvicorn app.api.main:app --reload --port 8000

# Terminal 2 — frontend, auto-reloads on TS/TSX changes
cd frontend && npm run dev
```

Open **http://localhost:5173** (Vite's dev server, not 8000). `frontend/vite.config.ts` proxies
`/api/*` and `/health` requests from 5173 to the backend on 8000, so the browser only ever talks to
one origin and CORS never comes into play in dev — the backend's `CORS_ORIGINS` allowlist
(`config.py`) exists only as a fallback for direct cross-origin requests.

### Single-command prod-like mode

```bash
cd frontend && npm run build && cd ..
.venv/bin/uvicorn app.api.main:app --port 8000
```

Open **http://localhost:8000** — with `frontend/dist/` present, `create_app()` mounts it as static
files at `/`, so one process serves both the API and the built UI (no Vite dev server, no proxy, no
CORS needed since it's all same-origin).

### Chat is one shared conversation

There's a single conversation held by the backend (`app/api/state.py`'s `AppState.chat_engine`), not
one per browser tab or session — this is a personal, single-user tool, so that's the simplest model
that fits. If you have two tabs open, they share one conversation; use **Clear chat** to start a fresh
topic rather than opening a second tab expecting a second conversation.

## Tests

```bash
.venv/bin/python -m pytest tests/                       # full suite (needs Ollama running)
.venv/bin/python -m pytest tests/ -m "not integration"   # skip tests that need a live Ollama server
cd frontend && npx tsc --noEmit                          # frontend type-check
```

## Troubleshooting

- **"Ollama unreachable" banner** — Ollama isn't running. Start it (`ollama serve`, or the desktop
  app), then click **Recheck** in the banner (or just wait — `/health` is polled automatically).
- **"Model(s) not pulled" banner** — Ollama is running but a model in `config.py`
  (`LLM_MODEL`/`EMBED_MODEL`) hasn't been pulled yet. Run the `ollama pull <model>` command the
  banner names, then **Recheck**.
- **Slow first answer / high RAM** — the first request after startup pays model-load latency; see
  [`docs/benchmarks.md`](./docs/benchmarks.md) for tokens/sec and peak RAM per model on this
  hardware, and switch `config.LLM_MODEL` to the smaller `llama3.2:3b` if `qwen2.5:7b` is too slow.
- **Answers seem to ignore retrieved context / get truncated** — check `config.CONTEXT_WINDOW`
  (currently 8192). Ollama's own default `num_ctx` is often smaller than a model's advertised max and
  will silently truncate retrieved chunks + chat history if the explicit `context_window` on the
  `Ollama(...)` LLM client is ever removed.
- **Upload rejected** — files over `config.MAX_UPLOAD_SIZE_BYTES` (50 MB) get a 413, and only
  `.pdf`/`.png`/`.jpg`/`.jpeg` are accepted (400 otherwise); both show up as a readable inline message,
  not a stack trace.

## Offline note

At runtime nothing needs the internet — Ollama serves models locally at `http://localhost:11434`, and
the vector store (Chroma) and manifest are plain local files under `data/`. Disable networking and the
whole upload → ask → answer flow still works, in either run mode above.
