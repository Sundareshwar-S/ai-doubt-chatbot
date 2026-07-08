// Mirrors app/api/schemas/*.py (Phase 5).

export interface DocumentEntry {
  source: string;
  sha256: string;
  pages: number;
  added_at: string;
}

export interface DocumentUploadResponse extends DocumentEntry {
  ingested: boolean;
}

export interface DocumentListResponse {
  documents: DocumentEntry[];
}

export interface ChatCitation {
  source: string;
  page: number;
  score: number;
}

export interface ChatAskResponse {
  answer: string;
  citations: ChatCitation[];
}

// Newline-delimited JSON events from POST /api/chat/stream (app/api/routers/chat.py).
export type ChatStreamEvent =
  | { type: 'token'; text: string }
  | { type: 'done'; citations: ChatCitation[] }
  | { type: 'error'; message: string };

export interface HealthResponse {
  status: 'ok' | 'degraded';
  detail: string | null;
}

export interface ErrorEnvelope {
  error: {
    code: string;
    message: string;
  };
}
