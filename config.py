"""Single source of settings for the AI Doubt Solver.

Phase 0 fixes the model + Ollama connection settings; index/retrieval settings
(chunk size, top_k, similarity cutoff, persist dir) are added in Phase 2.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# --- Ollama connection ---
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Models ---
# Default chat model chosen from the Phase 0 benchmark (see docs/benchmarks.md):
# llama3.2:3b -> 16.3 tok/s, 2.56 GB RAM (~2x faster & half the RAM of qwen2.5:7b).
# Quality fallback that also fits ~9 GB: "qwen2.5:7b" (7.9 tok/s, 5.06 GB).
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

# --- LLM call tuning (slow CPU inference) ---
REQUEST_TIMEOUT = 360.0  # seconds; CPU generation can be slow
# Explicit context window so Ollama doesn't silently truncate retrieved chunks + history
# (its default num_ctx is often only 2048-4096). Sized to fit top_k chunks + prompt.
CONTEXT_WINDOW = 8192

# --- Indexing / retrieval (Phase 2) ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 4
# Empirically tuned in Phase 3 (nomic-embed-text, single/two-doc test corpus):
# out-of-doc queries scored ~0.28-0.30, in-doc queries scored ~0.43-0.66 -- 0.2
# (the Phase 2 placeholder) never filtered anything. 0.35 sits in the gap.
SIMILARITY_CUTOFF = 0.35
PERSIST_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "documents"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# --- Chat memory (Phase 4) ---
# Caps retained multi-turn history so history + TOP_K retrieved chunks (~4*512 =
# 2k tokens) + prompt stay within CONTEXT_WINDOW (8192), keeping RAM/context
# bounded on CPU inference. NOTE: ChatMemoryBuffer counts tokens with tiktoken's
# cl100k encoding, not llama3.2's tokenizer, so this is an approximate budget with
# deliberate head-room; re-verify empirically if CHUNK_SIZE/TOP_K/this value grow.
CHAT_MEMORY_TOKEN_LIMIT = 1536

# --- API (Phase 5) ---
CORS_ORIGINS = ["http://localhost:5173"]  # Vite dev server origin only
API_HOST = "127.0.0.1"  # localhost-only; no auth, so never bind 0.0.0.0
API_PORT = 8000
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB; generous for scanned/image-heavy PDFs
UPLOADS_DIR = DATA_DIR / "uploads"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
HEALTH_CHECK_TIMEOUT_SECONDS = 5.0  # GET /health's ping to Ollama's /api/tags
