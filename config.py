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
# Default chat model: qwen2.5:3b. At ~the same speed/RAM as llama3.2:3b on CPU it follows
# grounding + summarization instructions noticeably more reliably (llama3.2:3b tended to
# over-refuse broad "what is this about" questions and paraphrase the exact refusal string).
# Drop-in fallbacks that also fit this machine: "llama3.2:3b" (faster benchmark, looser
# instruction-following) or "qwen2.5:7b" (higher quality, ~2x slower/RAM). See docs/benchmarks.md.
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"

# --- LLM call tuning (slow CPU inference) ---
REQUEST_TIMEOUT = 360.0  # seconds; CPU generation can be slow
# Explicit context window so Ollama doesn't silently truncate retrieved chunks + history
# (its default num_ctx is often only 2048-4096). Sized to fit top_k chunks + prompt.
CONTEXT_WINDOW = 8192
# Deterministic grounded answers: temperature 0 stops the model rambling/randomising over
# retrieved context (the Ollama client default is 0.75) and makes refusals reproducible in tests.
LLM_TEMPERATURE = 0.0
# Keep the model resident in Ollama between requests so each question skips cold model-load
# latency. Comfortable in 16 GB alongside a 3B model; Ollama accepts a duration string.
LLM_KEEP_ALIVE = "30m"

# --- Indexing / retrieval (Phase 2) ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 4
# Empirically tuned (nomic-embed-text). Measured top-node similarity by question type:
#   genuinely out-of-doc questions          ~0.25-0.30
#   broad "what is this about"/summary asks  ~0.35-0.37
#   focused in-doc questions                 ~0.43-0.66
# 0.30 sits in the gap so summary/meta questions still retrieve (the grounding prompt
# then summarizes them) while unrelated questions are dropped for a free, deterministic
# refusal. It was 0.35, which wrongly dropped "what is this pdf about" (0.348) and made
# whole-document summary questions fail. Re-verify empirically if the embedding model changes.
SIMILARITY_CUTOFF = 0.30
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
