# AI Doubt Solver (Local RAG) — Plan

A local, offline "ask doubts about your PDFs/images" tool built with RAG. This file covers
**Context → Architecture → Phased Plan → Verification**. The current-phase task checklist lives in
[`tasks.md`](./tasks.md).

---

## Context

**Problem / need.** Ask "doubts" (questions) about your own study material by uploading a PDF or an
image and getting answers grounded in that material — running **entirely offline on this laptop** for
privacy and zero cost. This is a classic Retrieval-Augmented Generation (RAG) "chat with your
documents" application.

**Hardware reality (drives every decision).** The machine reports as **AMD Ryzen 7 7730U** with
**Radeon Vega 8** integrated graphics (`lspci` → "Renoir/Cezanne"). Despite the "7000-series" name the
7730U is a **rebranded Zen 3** part, *not* a Zen 4 chip with an RDNA3 780M iGPU. Consequences:
- The Vega 8 iGPU is **not ROCm-supported**; the only GPU path is Vulkan, which gives marginal
  (sometimes negative) benefit on Vega 8. → **Treat this as a CPU-inference machine.**
- ~14 GB total RAM, ~9 GB free in practice. → **Small quantized models only (3B–8B, Q4).**

**Confirmed decisions.**
- Build style: **framework-assisted** (use LlamaIndex, don't reinvent retrieval).
- Image handling: **OCR / text path** (photos of notes, textbook pages, printed/handwritten problems).
  *No* visual/diagram understanding (no vision-language model).
- LLM location: **strictly local / offline.**
- Interface: **local web app in the browser.**
- Depth of "done": **solid personal tool** (reliable for daily personal use, not packaged for others).
- Documents: **persistent library** (saved on disk, queryable across restarts, can add more).
- Chat: **multi-turn with follow-ups** (history capped to protect RAM/speed).

**Intended outcome.** A `localhost` web app where you upload PDFs/images, which are OCR'd (when
needed), chunked, embedded, and stored in a persistent local vector DB; you then ask questions in a
multi-turn chat and get answers grounded in — and citing — your documents, with the model refusing to
answer when the documents don't support it. Everything runs offline on the 7730U.

---

## Requirements (testable)

**A. Document ingestion** — A1 upload PDF · A2 upload image (PNG/JPG) · A3 born-digital PDF → direct
text (no OCR) · A4 scanned PDF / image → OCR · A5 unparseable file → clear error, no crash · A6
multi-page PDFs retain page numbers for citations.

**B. Indexing & retrieval** — B1 chunk (configurable size/overlap) · B2 embed with a local CPU model
(no network) · B3 store chunks + embeddings + `{source, page}` in a local vector store · B4 retrieve
top-k for a question · B5 index persists to disk across restarts.

**C. Question answering** — C1 retrieved chunks + question → local LLM (Ollama) → answer · C2 answer
cites source file/page · C3 nothing relevant → "can't find in your documents" (no hallucination) · C4
works fully offline · C5 follow-up questions use prior turns as context.

**D. Web UI** — D1 localhost page: upload, see library, ask, read answer · D2 answers show citations ·
D3 progress/status during ingest & answering · D4 errors (bad file, model down) shown readably.

**E. Local / performance** — E1 CPU-only on the 7730U within ~9 GB RAM, no discrete GPU · E2 small
quantized model (3B–8B Q4) · E3 no internet at runtime · E4 a concrete latency target (fixed from the
Phase 0 benchmark) · E5 reproducible setup, documented.

**Out of scope (v1):** cloud/hybrid LLMs · visual understanding of diagrams/figures · multi-user /
auth · mobile / packaged native desktop · fine-tuning · large corpora & scaling · reliance on
GPU/ROCm · guaranteed handwriting/complex-math OCR (best-effort only) · document editing/annotation.

**Assumptions:** single user on `localhost`, no auth; domain-agnostic documents; English primary.

---

## Architecture

### Chosen stack (alternatives noted — vetoable)

| Concern | Chosen | Why | Alternatives / tradeoff |
|---|---|---|---|
| RAG framework | **LlamaIndex** | Best for pure document Q&A; simplest ingest→index→query→chat API | LangChain (heavier, agentic); hand-rolled (max control, most work) |
| LLM runtime | **Ollama** | Simplest local server, OpenAI-style API, first-class LlamaIndex integration, offline | llama.cpp / llama-server (faster Vulkan, more manual); LM Studio |
| Default model | **`llama3.2:3b` (Q4)** for responsiveness | Fast first token on CPU; good enough for grounded Q&A | **`qwen2.5:7b` (Q4)** quality option (~5.2 GB, ~8–12 tok/s CPU per 2026 benchmarks) — **final pick from the Phase 0 benchmark on this machine** |
| Embeddings | **`nomic-embed-text` via Ollama** | Best CPU size/quality balance, 8k context, one runtime (no torch) | `BAAI/bge-small-en-v1.5` via `HuggingFaceEmbedding` (strong, but large torch dependency) |
| Vector store | **Chroma (`PersistentClient`)** | Embedded, on-disk persistence (needed for the library), first-class LlamaIndex integration | LanceDB (embedded, scales further); FAISS (fastest, manual persistence) |
| PDF text | **PyMuPDF** | Fast, accurate born-digital text + page boundaries for citations | pdfplumber; Docling (better tables/layout, heavier) |
| OCR (images / scans) | **RapidOCR** | Best CPU speed/accuracy offline; pip-only, **no Tesseract binary** (PaddleOCR on ONNX Runtime) | pytesseract (needs system Tesseract); EasyOCR (heavier) |
| Chat memory | **LlamaIndex chat engine** (`condense_plus_context`) + `ChatMemoryBuffer` | Documented pattern for grounded multi-turn follow-ups, token-capped | Manual history stuffing (more code, easy to overflow context) |
| Backend API | **FastAPI** | `create_app()` factory; lifespan builds the index/chat-engine singleton once (replaces `st.cache_resource`); `Depends()`-based DI; thin routers over `app/index`/`app/qa` | Flask (no async, no Pydantic validation, no OpenAPI docs) |
| Frontend | **React + Vite** (TypeScript) | Fast dev server, HMR, minimal build config; plain hooks + `fetch` are enough state for this app's scope | Streamlit (no longer used — traded native widgets for a real API + custom UI); Gradio |
| Reranking | **Deferred** (optional later) | `SentenceTransformerRerank` improves retrieval but adds CPU latency | Enable later if retrieval quality needs it |

### System flow

```
       ┌────── React + Vite UI (localhost:5173 dev / same-origin prod) ───────┐
       │  upload PDF/image · library view · chat · citations · status banner  │
       └───────────┬────────────────────────────────────┬─────────────────────┘
                    │ fetch (Vite dev proxy, or same-origin prod build)
                    ▼
       ┌─────────────────────── FastAPI app (app/api/, :8000) ────────────────┐
       │  routers/documents.py         routers/chat.py       routers/health.py │
       │  POST/GET/DELETE /api/documents   POST /api/chat/ask, /reset  /health │
       └───────────┬───────────────────────────────┬───────────────────────────┘
       INGEST PATH │                                │       QUERY PATH
                    ▼                                ▼
     ┌───────────────────────────────┐          ┌──────────────────────────────┐
     │ ingest/extract.py             │          │ qa/chat.py  (multi-turn)      │
     │  PyMuPDF: born-digital text   │          │   ChatMemoryBuffer (capped)   │
     │  empty/scanned page? ──┐      │          └───────────────┬──────────────┘
     └────────────────────────│──────┘                          ▼
                              ▼                    ┌──────────────────────────────┐
                    ┌──────────────────┐           │ qa/engine.py                  │
                    │ ingest/ocr.py    │           │  retrieve top-k + grounding   │
                    │  RapidOCR (image │           │  prompt → Ollama LLM          │
                    │  / scanned page) │           │  → answer + source citations  │
                    └────────┬─────────┘           └───────────────┬──────────────┘
                             ▼  Documents(text + {source, page})   │ retrieve
                    ┌──────────────────────────────┐               │
                    │ index/pipeline.py            │               │
                    │  chunk → embed(nomic) → index │◄──────────────┘
                    └────────────┬─────────────────┘
                                 ▼
                    ┌──────────────────────────────┐   ┌─────────────────────────┐
                    │ index/store.py               │──▶│ Chroma (PersistentClient)│
                    │  Chroma vector store + LI     │   │  ./data/chroma (on disk) │
                    └──────────────────────────────┘   └─────────────────────────┘

   All LLM + embedding calls go to a local Ollama server (http://localhost:11434). No internet.
   app/api/main.py's lifespan builds the index + chat engine once at startup (the FastAPI
   equivalent of Streamlit's @st.cache_resource); ingestion writes straight into the same
   persisted Chroma collection the retriever queries, so no rebuild step is needed after uploads.
```

### File / module map (one responsibility per file)

```
ai-doubt-solver/
├─ README.md                 # setup (Ollama + venv + Node) and run instructions, troubleshooting
├─ requirements.txt          # pinned Python deps
├─ .gitignore                # ignore ./data, venv, caches, frontend/node_modules, frontend/dist
├─ config.py                 # single source of settings: model names, paths, chunk size, top_k,
│                             # CORS_ORIGINS, API_HOST/PORT, MAX_UPLOAD_SIZE_BYTES, UPLOADS_DIR,
│                             # FRONTEND_DIST_DIR
├─ scripts/
│  └─ benchmark.py           # Phase 0: measure tokens/sec + RAM per model; embed timing
├─ app/
│  ├─ __init__.py
│  ├─ ingest/
│  │  ├─ extract.py          # PDF born-digital text (PyMuPDF); routes scanned pages/images to OCR
│  │  └─ ocr.py              # RapidOCR wrapper: image/rendered-page → text
│  ├─ index/
│  │  ├─ store.py            # Chroma PersistentClient + ChromaVectorStore + StorageContext
│  │  └─ pipeline.py         # chunk → embed → VectorStoreIndex; add/insert/remove; writes manifest.json
│  ├─ qa/
│  │  ├─ engine.py           # retrieval + query engine; returns answer + source nodes (citations)
│  │  ├─ prompts.py          # grounding/system prompt templates ("answer only from context…")
│  │  └─ chat.py             # as_chat_engine(condense_plus_context) + ChatMemoryBuffer
│  └─ api/                   # FastAPI backend (replaces the old Streamlit UI)
│     ├─ main.py             # create_app(): lifespan builds index/chat-engine singleton once,
│     │                      # CORS, router registration, mounts frontend/dist as static (prod)
│     ├─ state.py            # AppState dataclass: handle, embed_model, index, chat_engine
│     ├─ dependencies.py     # Depends() getters reading from request.app.state
│     ├─ exceptions.py       # domain errors -> structured {error:{code,message}} JSON
│     ├─ schemas/            # documents.py, chat.py, health.py, errors.py (Pydantic models)
│     ├─ routers/            # documents.py, chat.py, health.py
│     └─ services/           # ingestion_service.py, qa_service.py, ollama_health.py
├─ frontend/                 # React + Vite (TypeScript) UI
│  ├─ package.json           # react, react-dom + vite, @vitejs/plugin-react, typescript
│  ├─ vite.config.ts         # dev proxy: /api, /health -> http://localhost:8000
│  └─ src/
│     ├─ App.tsx, main.tsx
│     ├─ api/                # client.ts (fetch wrapper), types.ts (mirrors Pydantic schemas)
│     ├─ hooks/               # useDocuments.ts, useChat.ts, useHealth.ts
│     └─ components/          # StatusBanner, UploadPanel, LibraryList, ChatWindow, ChatMessage
├─ data/                     # (gitignored) chroma/ vector DB · manifest.json · uploaded files
├─ tests/
│  ├─ fixtures/              # sample born-digital PDF, scanned PDF, and an image
│  ├─ test_extract.py
│  ├─ test_index.py
│  ├─ test_qa.py
│  ├─ test_chat.py
│  └─ test_api.py            # httpx.AsyncClient(ASGITransport) against create_app(),
│                             # dependency_overrides pointing at a tmp_path Chroma collection
└─ docs/
   └─ benchmarks.md          # Phase 0 results + chosen default model
```

### Verified integration facts (not invented)

- `from llama_index.llms.ollama import Ollama` (pkg `llama-index-llms-ollama`) — pass
  `request_timeout=360.0` (slow CPU calls) **and an explicit `context_window`** (see gotcha #1).
- `from llama_index.embeddings.ollama import OllamaEmbedding` (pkg `llama-index-embeddings-ollama`),
  `model_name="nomic-embed-text"`.
- `VectorStoreIndex`, `StorageContext`, `Settings` from `llama_index.core`;
  `SentenceSplitter(chunk_size=512, chunk_overlap=64)` from `llama_index.core.node_parser`.
- Chroma (pkg `llama-index-vector-stores-chroma`): `chromadb.PersistentClient(path=…)` →
  `client.get_or_create_collection(name)` → `ChromaVectorStore(chroma_collection=…)` →
  `StorageContext.from_defaults(vector_store=…)`. **Build:** `VectorStoreIndex(nodes,
  storage_context=…, embed_model=…)`. **Reload existing:** `VectorStoreIndex.from_vector_store(
  vector_store, storage_context=…, embed_model=…)`. **Add docs later:** `index.insert_nodes(nodes)`.
  (Confirmed by the official "Local RAG with Chroma + Ollama" cookbook.)
- Grounded refusal (C3): `index.as_query_engine(similarity_top_k=…, node_postprocessors=[
  SimilarityPostprocessor(similarity_cutoff=…)])` (`SimilarityPostprocessor` from
  `llama_index.core.postprocessor`) drops low-similarity hits; combine with an "answer only from the
  context, else say you can't find it" prompt. The response exposes `source_nodes` → citations.
- Multi-turn (C5): `from llama_index.core.memory import ChatMemoryBuffer` +
  `index.as_chat_engine(chat_mode="condense_plus_context",
  memory=ChatMemoryBuffer.from_defaults(token_limit=…))` (equivalently
  `CondensePlusContextChatEngine.from_defaults(retriever, memory=…, llm=…, context_prompt=…)` from
  `llama_index.core.chat_engine`). `condense_plus_context` rewrites the follow-up + history into a
  standalone question *before* retrieval — this is what makes "explain it / why is that?" resolve
  correctly. `chat_engine.reset()` clears memory. (All confirmed via LlamaIndex docs.)
- OCR: install **`rapidocr` + `onnxruntime`** (`pip install rapidocr onnxruntime`), then
  `from rapidocr import RapidOCR`. ⚠️ The older split package `rapidocr-onnxruntime` is **no longer
  the recommended install** per the RapidOCR docs — use `rapidocr`.
- Optional `SentenceTransformerRerank` from `llama_index.core.postprocessor` (deferred; adds latency).
- *To confirm when pulling in Phase 0:* exact Ollama tags (`llama3.2:3b`, `qwen2.5:7b`,
  `nomic-embed-text`).

### Key risks & mitigations

1. **Model speed/RAM on the 7730U** (biggest risk) → **Phase 0 benchmark first**; fall back to 3B or a
   smaller quant; keep only one model loaded.
2. **OCR accuracy on messy scans/handwriting** → best-effort only; RapidOCR default; allow a manual
   "paste/edit extracted text" escape hatch later.
3. **First-token latency / model load** → keep Ollama warm; `request_timeout=360`; `st.status` spinner.
4. **RAM contention with other apps** → cap chat memory (token buffer), one small model, README note.
5. **Hallucination / weak grounding** → "answer only from context; else say you can't find it" prompt
   + a similarity floor so irrelevant retrievals are dropped.
6. **Heavy dependencies** → prefer Ollama embeddings over HuggingFace to avoid a multi-GB torch pull.

### Review findings — verified gotchas to code around

Re-checked against current LlamaIndex, RapidOCR, and Streamlit docs. These are real pitfalls the plan
now accounts for (so implementation doesn't hit them as bugs):

1. **Ollama context window silently truncates.** Ollama's default `num_ctx` (often 2048–4096) can be
   smaller than the model's advertised max, silently dropping retrieved chunks + chat history and
   quietly degrading answers. → Set an explicit `context_window` on the `Ollama(...)` LLM sized to fit
   `top_k` chunks + capped history + prompt (start ~8192), and keep `chunk_size ≤ ~512` tokens so each
   chunk fits the embedder's input window comfortably. *(Phases 3–4.)*
2. **Streamlit re-runs the whole script on every interaction.** Anything not cached reloads each time.
   → Load the index / LLM / chat engine once via `@st.cache_resource`; keep the display transcript and
   per-message source citations in `st.session_state`; guard ingestion so the same uploaded file isn't
   re-embedded on every rerun (track processed file hashes); wire "Clear chat" to
   `chat_engine.reset()`. *(Phase 5.)*
3. **No built-in "list my library."** A vector-store-only index has no docstore to enumerate, so
   listing/removing documents isn't free. → Maintain a small `data/manifest.json` (filename, hash,
   pages, added-at) written at ingest time; use it for the library view and to delete a document's
   vectors by metadata filter in Chroma. *(Phases 2 & 5.)*
4. **Follow-up retrieval correctness.** `condense_plus_context` (chosen) already solves the "what does
   'it' refer to" problem by condensing history into a standalone retrieval query; naive
   history-stuffing would not. *(Phase 4 — this validates the design choice.)*
5. **RapidOCR package rename.** Use `rapidocr`, not the deprecated `rapidocr-onnxruntime`. *(Phase 1.)*

---

## Phased Plan

> Ordering rule: riskiest / most uncertain work first. Each phase lists Goal, Depends on, Files, and
> Definition of Done (DoD).

### Phase 0 — Environment & feasibility spike  *(riskiest first)*
- **Goal:** Prove this hardware can run the local stack at acceptable speed, and lock the model choice.
- **Depends on:** nothing.
- **Files:** `requirements.txt`, `README.md`, `scripts/benchmark.py`, `docs/benchmarks.md`, `config.py`.
- **DoD:** Ollama installed with candidate models pulled; `benchmark.py` produces a tokens/sec +
  peak-RAM table; a **default model is chosen** that fits ~9 GB and answers a short prompt within the
  agreed latency target; a minimal LlamaIndex↔Ollama smoke test answers a question **offline**.
- **Task checklist:** see [`tasks.md`](./tasks.md).

### Phase 1 — Document ingestion & extraction (PDF + image, OCR)
- **Goal:** Turn an uploaded PDF or image into clean text with `{source, page}` metadata.
- **Depends on:** Phase 0.
- **Files:** `app/ingest/extract.py`, `app/ingest/ocr.py`, `tests/test_extract.py`, `tests/fixtures/*`.
- **DoD:** born-digital PDF → direct text (no OCR); scanned PDF and a PNG/JPG → OCR'd text; page numbers
  retained; a corrupt/empty file raises a clear error (A5), not a crash.

### Phase 2 — Indexing, retrieval & persistence
- **Goal:** Chunk → embed → store in **persistent** Chroma; retrieve top-k; support a growing library.
- **Depends on:** Phase 1.
- **Files:** `app/index/store.py`, `app/index/pipeline.py`, `data/manifest.json`, `tests/test_index.py`.
- **DoD:** Ingest a doc, **restart the process**, and a query still returns relevant chunks with
  metadata; adding a second document works and both are searchable; a `data/manifest.json` backs
  list/remove (re-uploading the same file is a no-op, removing deletes its vectors).

### Phase 3 — RAG question answering (LLM + citations + grounding)
- **Goal:** Answer questions from retrieved chunks, with citations, refusing when unsupported.
- **Depends on:** Phase 2.
- **Files:** `app/qa/engine.py`, `app/qa/prompts.py`, `tests/test_qa.py`.
- **DoD:** in-document question → correct answer **with source/page citation** (C2); out-of-document
  question → "I can't find this in your documents" (C3); works offline (C4).

### Phase 4 — Multi-turn chat memory
- **Goal:** Follow-up questions use prior turns (C5) with a capped history.
- **Depends on:** Phase 3.
- **Files:** `app/qa/chat.py`, `tests/test_chat.py`.
- **DoD:** a follow-up like "explain that more simply" resolves using the previous turn; the memory
  buffer is token-capped so RAM stays bounded.

### Phase 5 — FastAPI backend + React/Vite frontend
- **Goal:** Browser app talking to a FastAPI API: upload, library view, chat with citations, live
  status, readable errors.
- **Depends on:** Phases 1–4.
- **Files:** `app/api/{main,state,dependencies,exceptions}.py`, `app/api/schemas/*`,
  `app/api/routers/*`, `app/api/services/*`, `frontend/*` (React + Vite).
- **DoD:** from the browser only, against the running FastAPI backend: upload → ask → cited answer →
  follow-up → "Clear chat" resets the conversation; ingest/answer show progress (D3, client-side
  spinners since responses are plain blocking JSON, no SSE); a bad file and a stopped-Ollama both
  surface a clear message via the `/health` banner and per-request error envelopes (D4). Works both
  in the two-process dev workflow (`uvicorn --reload` + `vite`) and the single-command prod mode
  (`uvicorn` alone serving the built `frontend/dist`).

### Phase 6 — Hardening & docs (solid personal tool)
- **Goal:** Make it reliable enough for daily personal use.
- **Depends on:** Phase 5.
- **Files:** `config.py`, `README.md`, `app/api/services/ingestion_service.py`,
  `app/api/services/ollama_health.py`, `frontend/src/components/StatusBanner.tsx`.
- **DoD:** a fresh setup works by following the README in **both** the dev (two-process) and
  single-command prod mode; a file-size guard and input validation live in the FastAPI upload
  endpoint; Ollama-not-running vs. model-missing are distinguished and recoverable without a
  restart (a "Recheck" action); settings live in `config.py`.

---

## Verification (overall)

- **Per phase:** each phase's DoD is the acceptance test; unit tests (`tests/`) accompany Phases 1–4
  using the sample fixtures.
- **Offline proof:** disable networking and confirm ingest + Q&A still work (Ollama is local).
- **End-to-end (dev):** `uvicorn app.api.main:app --reload --port 8000` in one terminal, `npm run dev`
  (Vite, port 5173) in another; upload a sample PDF and an image, ask an in-doc question (expect a
  cited answer), ask an out-of-doc question (expect a refusal), ask a follow-up (expect it to use
  prior context), then "Clear chat".
- **End-to-end (prod/daily-use):** `npm run build` in `frontend/`, then
  `uvicorn app.api.main:app --port 8000` alone serves both the API and the built UI at
  `http://localhost:8000` — repeat the same upload/ask/follow-up/clear walkthrough.
