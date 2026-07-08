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
  `pillow`, `numpy`, `streamlit`, `pytest`); add a `README.md` stub and `.gitignore` (ignore `data/`,
  `.venv/`, `__pycache__/`).
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

- [ ] **T2.1 — Index settings.** Add to `config.py`: `chunk_size=512`, `chunk_overlap=64`, `top_k`,
  `similarity_cutoff`, `persist_dir`, `collection_name`, `context_window`.
  *Files:* `config.py`.
  *Verify:* constants import cleanly.

- [ ] **T2.2 — Chroma store.** In `store.py`, build `chromadb.PersistentClient(path=persist_dir)` →
  `get_or_create_collection(collection_name)` → `ChromaVectorStore` → `StorageContext.from_defaults`.
  *Files:* `app/index/store.py`.
  *Verify:* calling it creates `data/chroma/`; returns store + storage_context.

- [ ] **T2.3 — Build/insert pipeline.** In `pipeline.py`, chunk Documents with
  `SentenceSplitter(chunk_size, chunk_overlap)`, embed with `OllamaEmbedding`, build
  `VectorStoreIndex(nodes, storage_context=…, embed_model=…)`; `add_documents(docs)` inserts via
  `index.insert_nodes(...)` for later additions.
  *Files:* `app/index/pipeline.py`.
  *Verify:* ingest a fixture → `chroma_collection.count() > 0`.

- [ ] **T2.4 — Reload across restart (B5).** Load an existing collection with
  `VectorStoreIndex.from_vector_store(vector_store, storage_context=…, embed_model=…)`.
  *Files:* `app/index/pipeline.py`, `app/index/store.py`.
  *Verify:* process A ingests; a **fresh process** B queries and gets a relevant chunk.

- [ ] **T2.5 — Library manifest.** Write/read `data/manifest.json` (filename, sha256, pages, added_at);
  ingest updates it and **skips re-ingest if the hash already exists**.
  *Files:* `app/index/pipeline.py`.
  *Verify:* ingest the same file twice → one manifest entry, no duplicate vectors.

- [ ] **T2.6 — List / remove.** `list_documents()` reads the manifest; `remove_document(source)` deletes
  the manifest entry **and** the doc's vectors via Chroma metadata filter on `source`.
  *Files:* `app/index/pipeline.py`.
  *Verify:* add two docs → list shows two; remove one → list shows one and its vectors are gone.

- [ ] **T2.7 — Retrieval test.** Retriever returns `top_k` nodes carrying `{source, page}` metadata.
  *Files:* `tests/test_index.py`.
  *Verify:* `pytest tests/test_index.py` passes; nodes carry metadata.

**Phase 2 DoD:** ingest → restart → query still works; second doc added & searchable; manifest-backed
list/remove works; `test_index.py` passes.

---

## Phase 3 — RAG question answering (LLM + citations + grounding)

- [ ] **T3.1 — Grounding prompt.** In `prompts.py`, a QA template: "Answer **only** from the provided
  context; if the answer isn't there, say you can't find it in the documents. Cite source + page."
  *Files:* `app/qa/prompts.py`.
  *Verify:* template loads and formats with `{context_str}` / `{query_str}`.

- [ ] **T3.2 — Query engine + citations (C2).** In `engine.py`, `index.as_query_engine(
  similarity_top_k=top_k, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=…)],
  text_qa_template=…)` with an `Ollama` LLM (`request_timeout=360`, `context_window`); format
  `response.source_nodes` into `{source} p.{page}` citations.
  *Files:* `app/qa/engine.py`.
  *Verify:* an in-doc question returns an answer **plus** ≥1 `{source, page}` citation.

- [ ] **T3.3 — Refusal path (C3).** With the cutoff, an out-of-doc question drops all hits → return the
  "can't find this in your documents" message, not a hallucination.
  *Files:* `app/qa/engine.py`, `app/qa/prompts.py`.
  *Verify:* an unrelated question → refusal message.

- [ ] **T3.4 — Offline check (C4).** With networking disabled, the engine still answers.
  *Files:* — (manual).
  *Verify:* disable net → query → answer returned.

- [ ] **T3.5 — QA tests.** In-doc → cited answer; out-of-doc → refusal.
  *Files:* `tests/test_qa.py`.
  *Verify:* `pytest tests/test_qa.py` passes.

**Phase 3 DoD:** cited answers for in-doc questions; refusal for out-of-doc; works offline;
`test_qa.py` passes.

---

## Phase 4 — Multi-turn chat memory

- [ ] **T4.1 — Chat engine.** In `chat.py`, `index.as_chat_engine(chat_mode="condense_plus_context",
  memory=ChatMemoryBuffer.from_defaults(token_limit=…), node_postprocessors=[SimilarityPostprocessor(
  similarity_cutoff=…)], llm=…)`.
  *Files:* `app/qa/chat.py`.
  *Verify:* `chat_engine.chat("…")` returns a response exposing `source_nodes`.

- [ ] **T4.2 — Follow-up resolution (C5).** Ask a question, then a pronoun follow-up ("explain that
  more simply"); confirm it uses the prior turn (condense step rewrites it to a standalone query).
  *Files:* `tests/test_chat.py`.
  *Verify:* follow-up answer stays on-topic; `pytest tests/test_chat.py` passes.

- [ ] **T4.3 — Memory cap + reset.** `token_limit` bounds history; `chat_engine.reset()` clears it.
  *Files:* `app/qa/chat.py`, `tests/test_chat.py`.
  *Verify:* after `reset()`, a follow-up no longer resolves against the prior turn.

**Phase 4 DoD:** follow-ups resolve via condensed history; memory is token-capped; reset works;
`test_chat.py` passes.

---

## Phase 5 — Streamlit web UI

- [ ] **T5.1 — App skeleton + caching.** In `streamlit_app.py`, load the index + chat engine once via
  `@st.cache_resource`; init `st.session_state.messages = []` behind the `if "messages" not in
  st.session_state` guard.
  *Files:* `app/ui/streamlit_app.py`.
  *Verify:* `streamlit run app/ui/streamlit_app.py` opens the page; the model isn't reloaded on each
  interaction (watch logs).

- [ ] **T5.2 — Upload & ingest (D3).** `st.file_uploader` (pdf/png/jpg) → run ingestion inside
  `st.status`/spinner; **guard against re-ingesting** the same file (hash check against the manifest).
  *Files:* `app/ui/streamlit_app.py`.
  *Verify:* upload a PDF → it appears in the library; re-running doesn't re-embed it.

- [ ] **T5.3 — Library view.** List documents from the manifest with a remove button per doc.
  *Files:* `app/ui/streamlit_app.py`.
  *Verify:* uploaded docs are listed; remove deletes the doc and its vectors.

- [ ] **T5.4 — Chat UI (D2).** `st.chat_input` / `st.chat_message`; replay transcript from
  `st.session_state`; render per-answer citations and **store the sources alongside each message** so
  prior turns keep their citations on rerun.
  *Files:* `app/ui/streamlit_app.py`.
  *Verify:* ask → cited answer renders; a follow-up works; earlier citations persist after rerun.

- [ ] **T5.5 — Error surfacing (D4).** Ollama-not-running and bad-file cases show a readable
  `st.error`, not a stack trace.
  *Files:* `app/ui/streamlit_app.py`.
  *Verify:* stop Ollama → friendly error; upload a garbage file → friendly error.

- [ ] **T5.6 — Clear chat.** Button → `chat_engine.reset()` + clear `st.session_state.messages`.
  *Files:* `app/ui/streamlit_app.py`.
  *Verify:* chat history and memory both reset.

**Phase 5 DoD:** from the browser only — upload → ask → cited answer → follow-up; progress shown;
errors friendly; no per-interaction model reload.

---

## Phase 6 — Hardening & docs (solid personal tool)

- [ ] **T6.1 — Centralize config.** All tunables (models, paths, chunk/top_k/cutoff/context_window,
  max file size) live in `config.py`; remove magic numbers from modules.
  *Files:* `config.py` + small edits across modules.
  *Verify:* grep shows no stray literals; app still runs.

- [ ] **T6.2 — Input validation + size guard.** Reject files over a size cap and unsupported types with
  a clear message.
  *Files:* `app/ingest/extract.py` and/or `app/ui/streamlit_app.py`.
  *Verify:* oversized/unsupported file → clean rejection, no crash.

- [ ] **T6.3 — Ollama health check.** On startup, ping `http://localhost:11434` / verify the chosen
  model is present; if down/missing, show actionable setup instructions.
  *Files:* `app/ui/streamlit_app.py` (or a small `app/health.py`).
  *Verify:* with Ollama stopped, the app shows instructions instead of failing.

- [ ] **T6.4 — README.** Full setup (install Ollama, pull models, venv, `pip install`, run) +
  troubleshooting (slow first token, RAM tips, `num_ctx`/context window) + the offline note.
  *Files:* `README.md`.
  *Verify:* following the README in a fresh shell brings the app up.

- [ ] **T6.5 — Final end-to-end + offline pass.** Walk the whole flow and fix rough edges.
  *Files:* — (whole app).
  *Verify:* the "Verification (overall)" checklist in `plan.md` passes, including the offline run.

**Phase 6 DoD:** fresh setup from README works; validation + health check + centralized config in
place; full end-to-end and offline passes are green.
