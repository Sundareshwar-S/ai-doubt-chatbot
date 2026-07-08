# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Local, fully offline RAG "chat with your PDFs/images" tool (AI Doubt Solver). Everything — LLM,
embeddings, vector store — runs on-machine via Ollama; no data or requests leave the host. The
project targets CPU inference, which drives model choice (`qwen2.5:3b`, quantized — chosen over
`llama3.2:3b` for tighter instruction-following/summarization at ~the same speed/RAM) and
context-window/chunk-size tuning throughout. Optional AMD iGPU offload (Vulkan/ROCm) is documented in
`README.md` but is a runtime/env setting only — no app code depends on it.

`plan.md` is the authoritative architecture/phased-plan doc; `tasks.md` is the current task
checklist. Read both before starting work on any phase — they contain grounded, verified API facts
(exact LlamaIndex/Chroma/RapidOCR call signatures) that should be reused rather than re-derived or
guessed.

## Commands

```bash
# Environment: pinned to Python 3.12 via uv (ML/OCR wheels — onnxruntime, chromadb, PyMuPDF —
# may not exist for newer Python releases). The venv has no pip binary; use `uv pip`.
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Ollama models (must be installed & running; default model set in config.py)
ollama pull qwen2.5:3b        # default chat model (config.LLM_MODEL)
ollama pull nomic-embed-text
ollama pull llama3.2:3b       # faster fallback, looser instruction-following
ollama pull qwen2.5:7b        # quality fallback, ~2x slower/RAM — see docs/benchmarks.md

# Tests
.venv/bin/python -m pytest tests/                     # full suite
.venv/bin/python -m pytest tests/test_qa.py -v         # single file
.venv/bin/python -m pytest tests/test_qa.py::test_out_of_doc_question_refuses  # single test
.venv/bin/python -m pytest tests/ -m "not integration" # skip tests needing a live Ollama server

# Phase 0 scripts (benchmark + offline proof)
.venv/bin/python scripts/benchmark.py   # tok/s + peak RAM per candidate model, embed speed
.venv/bin/python scripts/smoke.py       # index a paragraph, answer a question, no network needed
```

Most tests are marked `@pytest.mark.integration` and require a **real local Ollama server** at
`config.OLLAMA_BASE_URL` — they auto-skip (via the session-scoped `require_ollama` fixture in
`tests/conftest.py`) if it's unreachable, rather than mocking the LLM/embedding calls.

## File structure

```
├─ config.py               # single source of settings: models, Ollama URL, chunk/top_k/cutoff, paths
├─ requirements.txt
├─ pytest.ini               # registers the `integration` marker
├─ plan.md                  # architecture + phased plan (authoritative)
├─ tasks.md                 # per-phase task checklist
├─ README.md                # setup instructions
├─ app/
│  ├─ ingest/
│  │  ├─ extract.py         # PDF (PyMuPDF) + scanned-page routing -> Document list
│  │  └─ ocr.py             # RapidOCR wrapper (image -> text)
│  ├─ index/
│  │  ├─ store.py           # Chroma PersistentClient + ChromaVectorStore + StorageContext
│  │  ├─ pipeline.py        # chunk -> embed -> VectorStoreIndex; ingest/add/list/remove
│  │  └─ manifest.py        # data/manifest.json read/write (library list, dedup by sha256)
│  ├─ qa/
│  │  ├─ prompts.py         # grounding QA prompt template + REFUSAL_MESSAGE + chat context prompt
│  │  ├─ engine.py          # query engine, citations, grounded-refusal short-circuit
│  │  └─ chat.py            # multi-turn condense_plus_context chat engine + token-capped memory
│  └─ api/                  # FastAPI backend (Phase 5)
│     ├─ main.py            # create_app(); lifespan builds AppState once; CORS; static mount
│     ├─ state.py           # AppState: handle, embed_model, index, chat_engine, chat_lock, library_lock
│     ├─ dependencies.py    # Depends() getters reading from request.app.state (pure reads, no fallback)
│     ├─ exceptions.py      # domain errors -> {error:{code,message}} JSON envelope
│     ├─ schemas/           # documents.py, chat.py, health.py, errors.py (Pydantic models)
│     ├─ routers/           # documents.py, chat.py, health.py
│     └─ services/          # ingestion_service.py, qa_service.py, ollama_health.py
├─ scripts/
│  ├─ benchmark.py          # Phase 0: tok/s + peak RAM per candidate model
│  └─ smoke.py              # Phase 0: offline LlamaIndex<->Ollama smoke test
├─ docs/
│  └─ benchmarks.md         # Phase 0 results + chosen default model
├─ tests/
│  ├─ conftest.py           # `require_ollama` fixture + app_state/api_app/client fixtures (Phase 5)
│  ├─ fixtures/             # sample born-digital PDF, scanned PDF, image
│  ├─ test_extract.py, test_ocr.py, test_index.py, test_qa.py, test_chat.py, test_api.py, test_prompts.py
├─ frontend/                # React + Vite + TypeScript UI (Phase 5)
│  ├─ vite.config.ts        # dev proxy: /api, /health -> http://localhost:8000
│  └─ src/
│     ├─ App.tsx, main.tsx
│     ├─ api/               # client.ts (fetch wrapper + ApiError), types.ts (mirrors Pydantic schemas)
│     ├─ hooks/             # useDocuments.ts, useChat.ts, useHealth.ts
│     └─ components/        # StatusBanner, UploadPanel, LibraryList, ChatWindow, ChatMessage
└─ data/                    # (gitignored) chroma/ vector DB, manifest.json, uploaded files
```

## Architecture

Linear pipeline, one module per stage, each independently testable:

```
ingest/extract.py  →  index/pipeline.py  →  index/store.py (Chroma)  →  qa/engine.py
 (PDF/OCR → Documents)   (chunk → embed → persist)                       (retrieve → LLM → citations)
```

- **`app/ingest/extract.py`** — PyMuPDF for born-digital PDF text; a page is routed to OCR
  (`app/ingest/ocr.py`, RapidOCR) when its extracted text falls below
  `SCANNED_PAGE_TEXT_THRESHOLD`, rendered at `OCR_RENDER_DPI`. Produces `list[Document]` with
  `metadata={"source": filename, "page": n}` — this metadata shape is load-bearing: everything
  downstream (citations, manifest, delete-by-source) keys off `source`/`page`.
- **`app/index/store.py`** — `get_vector_store_handle()` opens/creates a persistent Chroma
  collection and returns a frozen `VectorStoreHandle` (client, collection, vector_store,
  storage_context). Tests pass an isolated `persist_dir`/`collection_name` (via `tmp_path`) instead
  of the real `data/chroma/`.
- **`app/index/pipeline.py`** — chunk (`SentenceSplitter`, `config.CHUNK_SIZE`/`CHUNK_OVERLAP`) →
  embed (`OllamaEmbedding`) → insert into the index. **`load_index()` always reloads live from the
  persisted Chroma collection** (`VectorStoreIndex.from_vector_store`) — it never trusts in-memory
  state. This is why a single index/chat-engine instance built once at process startup stays correct
  even after later uploads: ingestion writes straight into the same collection the retriever queries,
  no rebuild/invalidation step needed. `ingest_file()` dedups by content sha256 via
  `app/index/manifest.py` (`data/manifest.json`, atomic write via tempfile+`os.replace`) — a
  vector-store-only index has no docstore to enumerate, so the manifest is the only source of truth
  for "what's in the library" and backs list/remove.
- **`app/qa/engine.py`** — `answer_question()` builds a query engine
  (`SimilarityPostprocessor(similarity_cutoff=config.SIMILARITY_CUTOFF)` + a custom grounding
  `PromptTemplate` from `app/qa/prompts.py`) and returns a frozen `AnswerResult(answer, citations)`.
  **Refusal path:** when the cutoff drops every retrieved node, LlamaIndex's
  `BaseSynthesizer.synthesize()` already skips the LLM call entirely (returns early on
  `len(nodes) == 0`) — `answer_question` detects `response.source_nodes == []` and substitutes a
  fixed `REFUSAL_MESSAGE` for LlamaIndex's generic "Empty Response". This makes refusal both free
  (no wasted CPU generation) and deterministic to test (exact string match, not fuzzy LLM output).
  Citations are deduped by `(source, page)` keeping the max score, since one page can split into
  multiple retrieved chunks. `config.SIMILARITY_CUTOFF` was empirically tuned to **0.30**
  (with `nomic-embed-text`: genuinely out-of-doc queries score ~0.25-0.30, *broad "what is this
  about"/summary* questions ~0.35-0.37, focused in-doc ~0.43-0.66). It was 0.35, which dropped summary
  questions like "what is this pdf about" (0.348) *before the LLM ran* — a separate failure mode from
  the LLM over-refusing on retrieved context. 0.30 sits in the gap so meta questions retrieve (the
  prompt then summarizes) while unrelated ones still get the free deterministic refusal — re-verify
  empirically if the embedding model ever changes.
  `default_llm()` is the single place the `Ollama` client is built (both engine and chat import it);
  it sets `temperature=0` (`config.LLM_TEMPERATURE`) for deterministic grounded answers/reproducible
  refusals and `keep_alive` (`config.LLM_KEEP_ALIVE`) to keep the model warm in RAM between requests.
  **The grounding prompts (`app/qa/prompts.py`) deliberately permit summarizing/synthesizing across
  chunks and only refuse when the context is *unrelated* to the question** — an earlier "answer only if
  literally contained" wording made the small 3B model over-refuse broad "what is this about" questions;
  `tests/test_prompts.py` locks in that the `{context_str}`/`{query_str}` placeholders + `REFUSAL_MESSAGE`
  survive future edits.
- **`app/qa/chat.py`** — multi-turn chat via LlamaIndex `condense_plus_context`: each turn condenses
  the follow-up + prior history into a standalone retrieval query *before* retrieval, so pronoun
  follow-ups ("explain that more simply") resolve against earlier turns. Reuses `engine.py`'s
  `Citation`/`AnswerResult`/`dedup_citations`/`default_llm` (DRY) with the same
  `SimilarityPostprocessor` cutoff, plus a grounded `GROUNDED_CHAT_CONTEXT_PROMPT`. `default_memory()`
  builds a token-capped `ChatMemoryBuffer` (`config.CHAT_MEMORY_TOKEN_LIMIT`). **Refusal differs from
  the query engine:** the chat engine always writes to memory, so on empty retrieval LlamaIndex
  stores the literal "Empty Response" — `ask()` both returns `REFUSAL_MESSAGE` *and* repairs that
  poisoned assistant message in memory so it can't corrupt the next turn's condense input. The engine
  is stateful (one instance per conversation). `start_stream()` is the streaming counterpart: it calls
  `chat_engine.stream_chat()` (which sets `source_nodes` synchronously from retrieval and spawns a
  background history-writer thread), and returns a `StreamHandle` — either `refused=True` (empty
  retrieval: it drains the stream, repairs memory, and the caller emits `REFUSAL_MESSAGE`) or the live
  token generator + up-front deduped citations. Same refusal contract as `ask()`, just streamed.
- **`app/api/`** — FastAPI backend wrapping the pipeline above. `main.py`'s `lifespan` builds a single
  `AppState` (handle, embed_model, index, chat_engine) exactly once at startup — the FastAPI
  equivalent of `@st.cache_resource` — and mounts `frontend/dist` at `/` when it exists (after all
  routers, so it never shadows an API route). **`chat_engine` is one shared conversation, not one per
  session/tab** — a deliberate simplification for this single-user, no-auth tool; `AppState.chat_lock`
  serializes access to its mutable `ChatMemoryBuffer` so concurrent requests can't interleave writes,
  and `AppState.library_lock` does the same for the manifest+Chroma-collection mutations that upload/
  remove share. `dependencies.py`'s `get_state` is a pure read off `request.app.state` (no
  build-a-default fallback, unlike `app/index`/`app/qa`'s DI pattern) so "built once at startup" is
  structurally guaranteed. Blocking calls (PyMuPDF/OCR/embedding, `chat_engine.chat()`, file I/O) are
  always wrapped in `starlette.concurrency.run_in_threadpool` inside routers/services, never called
  directly from an `async def` route. Domain exceptions (`app/api/exceptions.py`) render as
  `{"error": {"code", "message"}}` via `register_exception_handlers`; a refusal is a normal 200
  `AnswerResult`, not routed through that machinery. **Chat has two endpoints:** `POST /api/chat/ask`
  returns the whole `AnswerResult` in one shot (`qa_service.ask_question`), while `POST /api/chat/stream`
  streams the same turn as newline-delimited JSON — `{"type":"token",…}` deltas then a terminal
  `{"type":"done","citations":[…]}` — via `qa_service.ask_question_stream`, an async generator that
  holds `chat_lock` for the whole stream and pulls the sync token generator through
  `run_in_threadpool(next, …)`. A mid-stream failure can't use the error envelope (200 already sent), so
  it surfaces as a terminal `{"type":"error",…}` event. `frontend/` (React + Vite + TS) talks to `/api/*`
  and `/health` via a shared `apiFetch()` client that throws a typed `ApiError` from that same
  envelope — except `useChat.ts` (raw `fetch` reading the NDJSON stream body for progressive rendering)
  and `useHealth.ts` (raw `fetch('/health')` since that endpoint's body is meaningful data on both 200
  and 503, not an error to throw).

**Config is centralized and flat** — `config.py` is the single source for model names, Ollama
connection settings, LLM call tuning (`LLM_TEMPERATURE`, `LLM_KEEP_ALIVE`, `CONTEXT_WINDOW`,
`REQUEST_TIMEOUT`), chunk/retrieval tuning, and paths. Don't hardcode any of these in `app/`;
extend `config.py` instead. Note the explicit `CONTEXT_WINDOW` on the `Ollama` LLM: Ollama's default
`num_ctx` is often smaller than the model's advertised max and will silently truncate retrieved
context otherwise.

**Dependency-injection pattern used throughout `app/index/` and `app/qa/`**: every function that
touches Ollama/Chroma accepts an optional handle/embed_model/llm/index parameter defaulting to a
`default_*()` constructor (e.g. `pipeline.default_embed_model()`, `engine.default_llm()`). Follow
this pattern for new code — it's what lets tests inject `tmp_path`-isolated instances instead of
touching `data/`.

**Phased build order** (see `plan.md`/`tasks.md` for full detail): Phase 0 (env/model choice) →
Phase 1 (ingestion/OCR) → Phase 2 (indexing/persistence) → Phase 3 (QA + citations + refusal) →
Phase 4 (multi-turn chat memory, `app/qa/chat.py`) → Phase 5 (FastAPI backend in `app/api/` +
React/Vite frontend in `frontend/`) → Phase 6 (hardening: input validation, health-check UX, README,
offline verification) are all implemented and tested.

## Git workflow

- All work happens on the **`develop`** branch. Commit changes there, not on `main`.
- `main` is reserved for stable/released snapshots only — do not commit directly to `main` unless the user explicitly asks for a release merge.
- After every commit to `develop`, push it to `origin/develop` right away — don't wait to be asked.
