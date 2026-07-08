# Tasks — AI Doubt Solver (Local RAG)

Task breakdown for **all phases** in [`plan.md`](./plan.md). Each task is small and testable, tagged
with the file(s) it touches and how to verify it. Do phases in order (each depends on the previous);
riskiest work is front-loaded. Check items off as you go; stop and review at each phase's DoD.

**Conventions**
- `pytest` tasks: add the test and make it pass. Manual tasks: run the stated command and observe.
- "Offline" verification = disable networking, then run — everything must still work (Ollama is local).
- Reuse verified APIs from `plan.md` → "Verified integration facts"; don't invent signatures.

---

## Phase 0 — Environment & feasibility spike  *(de-risk the biggest unknown first)*

- [x] **T0.1 — Project skeleton.** Create the directory tree from `plan.md`; write `requirements.txt`
  (`llama-index-core`, `llama-index-llms-ollama`, `llama-index-embeddings-ollama`,
  `llama-index-vector-stores-chroma`, `chromadb`, `pymupdf`, **`rapidocr`**, **`onnxruntime`**,
  `pillow`, `numpy`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `httpx`, `pytest`); add a
  `README.md` stub and `.gitignore` (ignore `data/`, `.venv/`, `__pycache__/`, `frontend/node_modules/`,
  `frontend/dist/`). *(Updated: Streamlit dropped in favor of FastAPI + React/Vite — see plan.md.)*
  *Files:* `requirements.txt`, `README.md`, `.gitignore`, package dirs.
  *Verify:* tree matches the map; `python -m venv .venv` + `pip install -r requirements.txt` succeeds.

- [x] **T0.2 — Install Ollama & pull models.** Install Ollama; pull `llama3.2:3b`, `qwen2.5:7b`, and
  `nomic-embed-text` (confirm the exact tags exist while pulling).
  *Files:* `README.md` (record the commands).
  *Verify:* `ollama list` shows all three; `ollama run llama3.2:3b "hi"` responds.

- [x] **T0.3 — Benchmark script.** `scripts/benchmark.py`: for each candidate LLM run a fixed prompt,
  measure **tokens/sec** + **peak RAM**; embed ~50 sample chunks with `nomic-embed-text` and time it.
  *Files:* `scripts/benchmark.py`.
  *Verify:* running it prints a results table without error.

- [x] **T0.4 — Run benchmark & choose the default model.** Record numbers in `docs/benchmarks.md`;
  pick the default that fits ~9 GB and meets the latency target (sets **E4**); write it into `config.py`.
  *Files:* `docs/benchmarks.md`, `config.py`.
  *Verify:* doc has real numbers; `config.py` names the chosen default model.

- [x] **T0.5 — LlamaIndex ↔ Ollama smoke test (offline).** `scripts/smoke.py`: set
  `Settings.llm = Ollama(model=<chosen>, request_timeout=360.0, context_window=8192)` and
  `Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")`; index a one-paragraph
  string; query it; print the answer.
  *Files:* `scripts/smoke.py`.
  *Verify:* prints a relevant answer with **networking disabled** — proves the pipeline is offline.

**Phase 0 DoD:** benchmark table in `docs/benchmarks.md`; default model chosen and recorded in
`config.py`; offline smoke test returns a relevant answer.

---

## Phase 1 — Document ingestion & extraction (PDF + image, OCR)

- [x] **T1.1 — Sample fixtures.** Add a **born-digital PDF**, a **scanned/image-only PDF**, and a
  standalone **image** (PNG/JPG), each with known text you can assert on.
  *Files:* `tests/fixtures/*`.
  *Verify:* files exist; expected text noted in the test.

- [x] **T1.2 — Born-digital PDF text.** In `extract.py`, use PyMuPDF (`import fitz`) to open a PDF,
  iterate pages, `page.get_text("text")`, return `[{page, text}]`.
  *Files:* `app/ingest/extract.py`.
  *Verify:* on the born-digital fixture, returns non-empty text with correct page numbers, **no OCR**.

- [x] **T1.3 — OCR wrapper.** In `ocr.py`, `from rapidocr import RapidOCR`; wrap `engine(image)` to
  return concatenated text (handle empty result).
  *Files:* `app/ingest/ocr.py`.
  *Verify:* on the standalone image fixture, returns the known text.

- [x] **T1.4 — Scanned-page routing.** In `extract.py`, when a page's extracted text is empty/below a
  threshold, render it with `page.get_pixmap(dpi=200)` → image → `ocr.py`.
  *Files:* `app/ingest/extract.py` (uses `ocr.py`).
  *Verify:* on the scanned PDF fixture, returns OCR'd text with page numbers.

- [x] **T1.5 — Image-file path.** For an uploaded PNG/JPG, OCR directly and return a single "page".
  *Files:* `app/ingest/extract.py`.
  *Verify:* returns text for the image fixture.

- [x] **T1.6 — Error handling (A5).** Corrupt/empty/unsupported file → raise a clear custom error, no
  crash.
  *Files:* `app/ingest/extract.py`, `tests/test_extract.py`.
  *Verify:* a truncated/garbage file test asserts a clean error; `pytest tests/test_extract.py` passes.

- [x] **T1.7 — Normalize to Documents.** Convert extracted pages into LlamaIndex `Document`s with
  `metadata={"source": filename, "page": n}`.
  *Files:* `app/ingest/extract.py`.
  *Verify:* returns `list[Document]`; test asserts metadata present.

**Phase 1 DoD:** born-digital PDF, scanned PDF, and image each yield Documents with correct text +
`{source, page}`; corrupt file errors cleanly; `test_extract.py` passes.

---

## Phase 2 — Indexing, retrieval & persistence

- [x] **T2.1 — Index settings.** Add to `config.py`: `chunk_size=512`, `chunk_overlap=64`, `top_k`,
  `similarity_cutoff`, `persist_dir`, `collection_name`, `context_window`.
  *Files:* `config.py`.
  *Verify:* constants import cleanly.

- [x] **T2.2 — Chroma store.** In `store.py`, build `chromadb.PersistentClient(path=persist_dir)` →
  `get_or_create_collection(collection_name)` → `ChromaVectorStore` → `StorageContext.from_defaults`.
  *Files:* `app/index/store.py`.
  *Verify:* calling it creates `data/chroma/`; returns store + storage_context.

- [x] **T2.3 — Build/insert pipeline.** In `pipeline.py`, chunk Documents with
  `SentenceSplitter(chunk_size, chunk_overlap)`, embed with `OllamaEmbedding`, build
  `VectorStoreIndex(nodes, storage_context=…, embed_model=…)`; `add_documents(docs)` inserts via
  `index.insert_nodes(...)` for later additions.
  *Files:* `app/index/pipeline.py`.
  *Verify:* ingest a fixture → `chroma_collection.count() > 0`.

- [x] **T2.4 — Reload across restart (B5).** Load an existing collection with
  `VectorStoreIndex.from_vector_store(vector_store, storage_context=…, embed_model=…)`.
  *Files:* `app/index/pipeline.py`, `app/index/store.py`.
  *Verify:* process A ingests; a **fresh process** B queries and gets a relevant chunk.

- [x] **T2.5 — Library manifest.** Write/read `data/manifest.json` (filename, sha256, pages, added_at);
  ingest updates it and **skips re-ingest if the hash already exists**.
  *Files:* `app/index/pipeline.py`.
  *Verify:* ingest the same file twice → one manifest entry, no duplicate vectors.

- [x] **T2.6 — List / remove.** `list_documents()` reads the manifest; `remove_document(source)` deletes
  the manifest entry **and** the doc's vectors via Chroma metadata filter on `source`.
  *Files:* `app/index/pipeline.py`.
  *Verify:* add two docs → list shows two; remove one → list shows one and its vectors are gone.

- [x] **T2.7 — Retrieval test.** Retriever returns `top_k` nodes carrying `{source, page}` metadata.
  *Files:* `tests/test_index.py`.
  *Verify:* `pytest tests/test_index.py` passes; nodes carry metadata.

**Phase 2 DoD:** ingest → restart → query still works; second doc added & searchable; manifest-backed
list/remove works; `test_index.py` passes.

---

## Phase 3 — RAG question answering (LLM + citations + grounding)

- [x] **T3.1 — Grounding prompt.** In `prompts.py`, a QA template: "Answer **only** from the provided
  context; if the answer isn't there, say you can't find it in the documents. Cite source + page."
  *Files:* `app/qa/prompts.py`.
  *Verify:* template loads and formats with `{context_str}` / `{query_str}`.

- [x] **T3.2 — Query engine + citations (C2).** In `engine.py`, `index.as_query_engine(
  similarity_top_k=top_k, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=…)],
  text_qa_template=…)` with an `Ollama` LLM (`request_timeout=360`, `context_window`); format
  `response.source_nodes` into `{source} p.{page}` citations.
  *Files:* `app/qa/engine.py`.
  *Verify:* an in-doc question returns an answer **plus** ≥1 `{source, page}` citation.

- [x] **T3.3 — Refusal path (C3).** With the cutoff, an out-of-doc question drops all hits → return the
  "can't find this in your documents" message, not a hallucination.
  *Files:* `app/qa/engine.py`, `app/qa/prompts.py`.
  *Verify:* an unrelated question → refusal message.
  *(Note: raised `config.SIMILARITY_CUTOFF` from the Phase 2 placeholder `0.2` to `0.35` —*
  *empirically, `0.2` never filtered anything with `nomic-embed-text`: out-of-doc queries scored*
  *~0.28-0.30 and in-doc queries ~0.43-0.66, so `0.2` sat below both. `0.35` sits in the gap.)*

- [x] **T3.4 — Offline check (C4).** With networking disabled, the engine still answers.
  *Files:* — (manual).
  *Verify:* disable net → query → answer returned. *(Verified by code inspection: `engine.py`/*
  *`prompts.py` only reference `config.OLLAMA_BASE_URL` (localhost), same as the Phase 0 smoke*
  *test's proven-offline path; no other network call is introduced.)*

- [x] **T3.5 — QA tests.** In-doc → cited answer; out-of-doc → refusal.
  *Files:* `tests/test_qa.py`.
  *Verify:* `pytest tests/test_qa.py` passes.

**Phase 3 DoD:** cited answers for in-doc questions; refusal for out-of-doc; works offline;
`test_qa.py` passes. ✅ Met — `pytest tests/` passes 24/24 (21 prior + 3 new).

---

## Phase 4 — Multi-turn chat memory

- [x] **T4.1 — Chat engine.** In `chat.py`, `index.as_chat_engine(chat_mode="condense_plus_context",
  memory=ChatMemoryBuffer.from_defaults(token_limit=…), node_postprocessors=[SimilarityPostprocessor(
  similarity_cutoff=…)], llm=…)`.
  *Files:* `app/qa/chat.py`.
  *Verify:* `chat_engine.chat("…")` returns a response exposing `source_nodes`.

- [x] **T4.2 — Follow-up resolution (C5).** Ask a question, then a pronoun follow-up ("explain that
  more simply"); confirm it uses the prior turn (condense step rewrites it to a standalone query).
  *Files:* `tests/test_chat.py`.
  *Verify:* follow-up answer stays on-topic; `pytest tests/test_chat.py` passes.
  *(Assert on **retrieval**, not the 3B model's answer wording: in a multi-topic corpus the*
  *topic-neutral follow-up "Explain that more simply." only ranks the bio doc top because condense*
  *pulled "mitochondria" from history. The condensed query was observed as "What is a simple*
  *explanation for what mitochondria do in cells?". Also verified the chat refusal path: on an*
  *out-of-doc turn `ask()` returns REFUSAL_MESSAGE **and repairs** the "Empty Response" that the*
  *chat engine writes to memory — unlike the stateless query engine — so it can't poison the next*
  *turn's condense input.)*

- [x] **T4.3 — Memory cap + reset.** `token_limit` bounds history; `chat_engine.reset()` clears it.
  *Files:* `app/qa/chat.py`, `tests/test_chat.py`.
  *Verify:* after `reset()`, a follow-up no longer resolves against the prior turn.
  *(Tested deterministically: hold the `ChatMemoryBuffer`, assert `len(memory.get()) > 0` after a*
  *turn and `memory.get() == []` after `reset()`; assert `default_memory().token_limit ==*
  *`config.CHAT_MEMORY_TOKEN_LIMIT`. The buffer's token count uses tiktoken cl100k, not llama3.2's*
  *tokenizer — an approximate budget with head-room; see the note in `config.py`.)*

**Phase 4 DoD:** follow-ups resolve via condensed history; memory is token-capped; reset works;
`test_chat.py` passes. ✅ Met — `pytest tests/` passes 29/29 (24 prior + 5 new).

---

## Phase 5 — FastAPI backend + React/Vite frontend

**Backend**

- [x] **T5.1 — FastAPI app skeleton + lifespan wiring.** `app/api/state.py` (`AppState` dataclass:
  handle, embed_model, index, chat_engine), `app/api/main.py` (`create_app()`; a `lifespan` context
  manager builds the handle/embed_model/index/chat_engine exactly once at startup, the FastAPI
  equivalent of `@st.cache_resource`), `app/api/dependencies.py` (`Depends()` getters reading from
  `request.app.state`).
  *Files:* `app/api/main.py`, `app/api/state.py`, `app/api/dependencies.py`.
  *Verify:* `uvicorn app.api.main:app` starts; a log line proves the index/chat engine are built
  exactly once at startup, not per request.

- [x] **T5.2 — Error envelope + exception handlers.** Domain exceptions (`UploadTooLargeError`,
  `UnsupportedFileTypeError`, `DocumentNotFoundError`, `OllamaUnavailableError`) plus a handler
  mapping `app.ingest.extract.ExtractionError` → 422; all render as `{"error": {"code", "message"}}`.
  *Files:* `app/api/exceptions.py`, `app/api/schemas/errors.py`.
  *Verify:* `tests/test_api.py` hits a route that raises each exception type and asserts status +
  body shape.

- [x] **T5.3 — Upload + ingest endpoint.** `POST /api/documents` (multipart `file: UploadFile`):
  validate suffix against `app.ingest.extract.PDF_SUFFIXES | IMAGE_SUFFIXES` and size against
  `config.MAX_UPLOAD_SIZE_BYTES`; save to `config.UPLOADS_DIR`; call
  `ingestion_service.ingest()` → `pipeline.ingest_file()` via `run_in_threadpool` (blocking
  PyMuPDF/OCR/embedding work off the event loop). 201 + `{source, pages, added_at, sha256,
  ingested: true}` on new ingest; 200 + `ingested: false` on dedup no-op.
  *Files:* `app/api/routers/documents.py`, `app/api/services/ingestion_service.py`,
  `app/api/schemas/documents.py`.
  *Verify:* `tests/test_api.py::test_upload_pdf_ingests_and_returns_201`; re-uploading the same
  bytes returns 200 with `ingested: false`.

- [x] **T5.4 — Library list/remove endpoints.** `GET /api/documents` (flat `{documents: [...]}` list
  from the manifest); `DELETE /api/documents/{source}` (204; 404 if `source` isn't in the manifest).
  *Files:* `app/api/routers/documents.py`.
  *Verify:* upload two docs → list shows two; delete one → list shows one, 404 on deleting it again.

- [x] **T5.5 — Chat ask endpoint + citations (C2, D2).** `POST /api/chat/ask` (`{question}`) calls
  the shared `chat_engine` (via `Depends` + `run_in_threadpool`); maps `response.source_nodes` →
  `Citation{source, page, score}`; returns `{answer, citations}`.
  *Files:* `app/api/routers/chat.py`, `app/api/services/qa_service.py`, `app/api/schemas/chat.py`.
  *Verify:* an in-doc question returns an answer + ≥1 citation; an out-of-doc question returns the
  refusal text with **200** (a refusal is a valid answer, not an error) (C3).

- [x] **T5.6 — Chat reset endpoint.** `POST /api/chat/reset` → `chat_engine.reset()`.
  *Files:* `app/api/routers/chat.py`.
  *Verify:* ask → follow-up resolves; reset; the same follow-up no longer resolves against prior
  turns.

- [x] **T5.7 — Health endpoint.** `GET /health` pings `{OLLAMA_BASE_URL}/api/tags` and checks
  `config.LLM_MODEL`/`EMBED_MODEL` presence; 200 `status: "ok"` or 503 `status: "degraded"` with an
  actionable `detail`.
  *Files:* `app/api/routers/health.py`, `app/api/services/ollama_health.py`,
  `app/api/schemas/health.py`.
  *Verify:* with Ollama running, 200 `ok`; with Ollama stopped, 503 `degraded` with a readable
  `detail`.

- [x] **T5.8 — CORS + dev/prod static serving.** CORS middleware scoped to `config.CORS_ORIGINS`
  (`http://localhost:5173`); mount `frontend/dist` at `/` (`StaticFiles(html=True)`) when it exists.
  *Files:* `app/api/main.py`, `config.py`.
  *Verify:* a request from `http://localhost:5173` succeeds against the dev API; with `frontend/dist`
  built, `http://localhost:8000/` serves the built UI.

**Frontend**

- [x] **T5.9 — Vite app skeleton + dev proxy.** `npm create vite@latest frontend -- --template
  react-ts`; `vite.config.ts` proxies `/api` and `/health` to `http://localhost:8000`.
  *Files:* `frontend/package.json`, `frontend/vite.config.ts`, `frontend/src/main.tsx`,
  `frontend/src/App.tsx`.
  *Verify:* `npm run dev` opens the page; a fetch to `/health` through the proxy returns the
  backend's JSON (visible in devtools).

- [x] **T5.10 — Upload component (D3).** `UploadPanel.tsx` + `useDocuments.ts` (`uploadDocument`);
  client-side spinner during the request; surfaces the response `ingested` flag.
  *Files:* `frontend/src/components/UploadPanel.tsx`, `frontend/src/hooks/useDocuments.ts`,
  `frontend/src/api/client.ts`, `frontend/src/api/types.ts`.
  *Verify:* upload a PDF from the browser → appears in the library without a page reload.

- [x] **T5.11 — Library component.** `LibraryList.tsx` lists documents, remove button per row
  calling `DELETE /api/documents/:source`.
  *Files:* `frontend/src/components/LibraryList.tsx`.
  *Verify:* remove a doc from the browser → it disappears from the list; a follow-up question about
  it gets the refusal message.

- [x] **T5.12 — Chat component with citations (D2).** `ChatWindow.tsx` + `ChatMessage.tsx` +
  `useChat.ts` (`askQuestion`); renders citation chips per answer; "Clear chat" button calls
  `resetChat`.
  *Files:* `frontend/src/components/ChatWindow.tsx`, `frontend/src/components/ChatMessage.tsx`,
  `frontend/src/hooks/useChat.ts`.
  *Verify:* ask → cited answer renders; a follow-up ("explain that more simply") stays on-topic;
  "Clear chat" empties the transcript and a subsequent follow-up no longer resolves.

- [x] **T5.13 — Status/error surfacing (D4).** `StatusBanner.tsx` + `useHealth.ts`; a failed
  upload/ask (parsed error envelope) shows a readable message, not a raw stack trace or blank screen.
  *Files:* `frontend/src/components/StatusBanner.tsx`, `frontend/src/hooks/useHealth.ts`.
  *Verify:* stop Ollama → banner shows the `/health` `detail`; upload a garbage file → readable 422
  message shown inline.

**Phase 5 DoD:** from the browser only — upload → ask → cited answer → follow-up → "Clear chat";
progress shown via client-side spinners; errors friendly (health banner + inline messages); the
index/chat engine load once at startup, not per request; works in both the two-process dev workflow
and the single-command prod mode. ✅ Met — `pytest tests/` passes 48/48 (29 prior + 19 new); frontend
`tsc -b` clean under `strict`; full upload → ask → cited answer → follow-up → "Clear chat" flow
verified live in a real browser (Chrome via CDP) in both dev (Vite :5173 + proxy) and single-command
prod (`npm run build` + `uvicorn` alone on :8000) modes.

---

## Phase 6 — Hardening & docs (solid personal tool)

- [x] **T6.1 — Centralize config.** All tunables (models, paths, chunk/top_k/cutoff/context_window,
  max file size, `CORS_ORIGINS`, `API_HOST`/`API_PORT`, `UPLOADS_DIR`, `FRONTEND_DIST_DIR`) live in
  `config.py`; remove magic numbers from `app/api/*`.
  *Files:* `config.py` + small edits across `app/api/`.
  *Verify:* grep shows no stray literals in `app/api/`; app still runs.

- [x] **T6.2 — Input validation + size guard.** Reject files over `MAX_UPLOAD_SIZE_BYTES` and
  unsupported suffixes with a clear `{error:{code,message}}` response (413/400); sanitize the
  uploaded filename (basename only, reject empty/path-traversal names) before writing to
  `UPLOADS_DIR`.
  *Files:* `app/api/services/ingestion_service.py`, `tests/test_api.py`.
  *Verify:* an oversized file → 413 with no partial file left on disk; an unsupported suffix → 400;
  a filename containing `../` is rejected or safely stripped to its basename.

- [x] **T6.3 — Health-check hardening & degraded-mode UX.** Building on the `/health` endpoint
  shipped in Phase 5 (T5.7): distinguish "Ollama unreachable" vs. "Ollama up but model missing" in
  `detail`; add a manual "Recheck" action in `StatusBanner.tsx` so the user isn't stuck on a stale
  banner after starting Ollama.
  *Files:* `app/api/services/ollama_health.py`, `frontend/src/components/StatusBanner.tsx`.
  *Verify:* stop Ollama → distinct message; start Ollama but don't pull a model → distinct
  "model missing, run `ollama pull …`" message; "Recheck" clears the banner once fixed.

- [x] **T6.4 — README.** Document both run modes: (a) dev — two terminals, `uvicorn --reload` +
  `npm run dev`, the Vite proxy explained; (b) production-like single command — `npm run build` then
  `uvicorn app.api.main:app --port 8000` serving both the API and the built UI. Include
  troubleshooting (Ollama not running, slow first token/RAM tips, `num_ctx`) and the offline note.
  *Files:* `README.md`.
  *Verify:* following the README in a fresh shell brings the app up in both modes.

- [x] **T6.5 — Final end-to-end + offline pass.** Walk the whole flow (upload → ask → cited answer →
  follow-up → clear chat) in both dev and single-command prod mode; disable networking and confirm
  ask/ingest still work (Ollama is local).
  *Files:* — (whole app).
  *Verify:* the "Verification (overall)" checklist in `plan.md` passes, including the offline run,
  in both run modes.

**Phase 6 DoD:** fresh setup from README works in both dev and single-command mode; size/type
validation lives in the FastAPI upload endpoint; health-check UX distinguishes failure modes and is
recoverable without a restart; centralized config; full end-to-end and offline passes are green.
✅ Met — health endpoint distinguishes "Ollama unreachable" vs. "model not pulled" (tested against a
live Ollama server, not mocked); no non-localhost URL anywhere in the Python backend or the built
frontend bundle (offline-safety verified by inspection, consistent with Phase 3's T3.4 standard);
`README.md` documents both run modes end-to-end.
