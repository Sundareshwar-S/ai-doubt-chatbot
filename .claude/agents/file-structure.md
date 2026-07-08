---
name: file-structure
description: File structure specialist for the project. Use PROACTIVELY when creating new files or directories.
tools: ["Read", "Write", "Edit", "Bash", "Grep", "Glob"]
---

├── docker-compose.yml
├── .env
├── .env.example
├── .gitignore
├── .python-version         (Python 3.11 for local dev / tooling)
├── plan.md                 (step-by-step AI agent execution plan for building the project)
├── README.md
├── docs/
│   └── GUIDES/
│       └── architecture-and-workflow.md   (end-to-end pipeline, models, flowcharts)
├── .cursor/
│   ├── agents/             (Cursor subagent definitions — architect, planner, file-structure, code-reviewer, etc.)
│   ├── skills/             (project skills — fastapi-patterns, python-testing, iterative-retrieval, docker-patterns, etc.)
│   └── rules/
│       └── python/         (Python coding style, testing, security, FastAPI, hooks conventions)
├── scripts/
│   └── start-ollama.sh     (starts Ollama with OLLAMA_MODELS=./ollama_models/models)
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_ingest.py
│   │   ├── routes_query.py
│   │   └── routes_repos.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── git_cloner.py
│   │   ├── file_walker.py
│   │   ├── text_probe.py         (detect text-readable files; byte-capped reads)
│   │   ├── ast_chunker.py
│   │   ├── notebook_chunker.py   (extract code/markdown cells from .ipynb)
│   │   ├── embedding_service.py
│   │   ├── embed_cache.py       (content-hash cache for embedding vectors)
│   │   ├── manifest_builder.py  (file-tree manifest chunks for structure retrieval)
│   │   ├── chroma_writer.py
│   │   └── bm25_builder.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── sparse_retriever.py
│   │   ├── file_target_retriever.py   (deterministic retrieval when query names a file)
│   │   ├── structure_retriever.py     (directory listing / file-structure queries)
│   │   ├── query_intent.py            (shared intent detection for retrieval & prompts)
│   │   ├── rrf_fusion.py
│   │   ├── reranker.py
│   │   ├── prompt_builder.py
│   │   └── ollama_client.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   ├── limiter.py           (shared slowapi rate-limiter instance)
│   │   ├── debug_log.py         (NDJSON debug logging for Cursor debug sessions)
│   │   └── logger.py
│   └── jobs/
│       ├── __init__.py
│       └── job_store.py
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── public/
│   │   └── favicon.ico
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── components/
│       │   ├── Sidebar/
│       │   │   ├── RepoManager.jsx
│       │   │   ├── IngestionProgress.jsx
│       │   │   └── IndexedRepoList.jsx
│       │   ├── Chat/
│       │   │   ├── ChatWindow.jsx
│       │   │   ├── UserMessage.jsx
│       │   │   ├── AssistantMessage.jsx
│       │   │   ├── StreamingCursor.jsx
│       │   │   └── QueryInput.jsx
│       │   └── Sources/
│       │       ├── SourceCards.jsx
│       │       └── CodeSnippetCard.jsx
│       ├── hooks/
│       │   ├── useStreamingChat.js
│       │   ├── useIngestion.js
│       │   └── useRepos.js
│       ├── services/
│       │   └── api.js
│       └── styles/
│           └── index.css
├── tests/
│   ├── golden_qa/
│   │   ├── README.md
│   │   └── markupsafe_qa.json     (golden Q&A fixtures for eval)
│   ├── conftest.py
│   ├── test_api_query.py
│   ├── test_doc_chunking.py
│   ├── test_embed_performance.py
│   ├── test_embedding_ingest.py
│   ├── test_file_target_retriever.py
│   ├── test_file_walker.py
│   ├── test_notebook_chunking.py
│   ├── test_orchestrator_limits.py
│   ├── test_phase1_integration.py
│   ├── test_phase2_retrieval.py
│   ├── test_phase3_e2e.py
│   ├── test_structure_retriever.py
│   ├── test_text_probe.py
│   ├── eval_retrieval.py
│   ├── eval_generation.py
│   └── load_test.py
├── chroma_data/           (gitignored — Docker volume)
├── bm25_cache/            (gitignored — Docker volume, persists BM25 indexes across restarts)
├── ollama_models/         (gitignored — local + Docker model storage)
│   ├── models/            (blobs + manifests; set OLLAMA_MODELS to this path)
│   └── ollama.log         (gitignored — local serve log)
└── repos_cache/           (gitignored — ephemeral clones)

## Placement guidelines

- **API routes** → `backend/api/`
- **Ingestion pipeline** → `backend/ingestion/`
- **Retrieval / RAG** → `backend/retrieval/`
- **Shared config / deps** → `backend/core/`
- **Pydantic schemas** → `backend/models/`
- **Background jobs** → `backend/jobs/`
- **React components by feature** → `frontend/src/components/{Sidebar,Chat,Sources}/`
- **Hooks** → `frontend/src/hooks/`
- **Tests** → `tests/` (unit/integration naming: `test_*.py`)
- **Eval scripts** → `tests/eval_*.py`
