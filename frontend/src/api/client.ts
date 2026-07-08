// Shared fetch wrapper: every hook (useDocuments, useChat) parses errors the
// same way, from the backend's `{"error": {"code", "message"}}` envelope
// (app/api/exceptions.py). `/health` is deliberately NOT routed through this
// -- its body is meaningful application data on both 200 and 503, not an
// error to throw (see useHealth.ts).
import type { ErrorEnvelope } from './types';

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (response.status === 204) {
    return undefined as T;
  }

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    const envelope = body as ErrorEnvelope | null;
    const code = envelope?.error?.code ?? 'unknown_error';
    const message =
      envelope?.error?.message ?? `Request failed with status ${response.status}.`;
    throw new ApiError(response.status, code, message);
  }

  return body as T;
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError(0, 'network_error', 'Could not reach the server.');
  }
  return parseResponse<T>(response);
}
