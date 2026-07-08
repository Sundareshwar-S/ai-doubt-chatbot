// Polls GET /health so a stopped/started Ollama server is reflected without a
// page reload; `refetch` also backs StatusBanner's manual "Recheck" button
// (T6.3) -- same function, no separate rewrite needed.
//
// Deliberately bypasses api/client.ts's apiFetch/ApiError: /health's body
// (`{status, detail}`) is meaningful application data on *both* 200 and 503,
// unlike every other endpoint where a non-2xx means "throw the error
// envelope" -- so this hook parses the body regardless of status code.
import { useCallback, useEffect, useState } from 'react';
import type { HealthResponse } from '../api/types';

const POLL_INTERVAL_MS = 15000;

export function useHealth() {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const refetch = useCallback(async () => {
    try {
      const response = await fetch('/health');
      const body = (await response.json()) as HealthResponse;
      setHealth(body);
    } catch {
      setHealth({ status: 'degraded', detail: 'Could not reach the server.' });
    }
  }, []);

  useEffect(() => {
    refetch();
    const id = setInterval(refetch, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [refetch]);

  return { health, refetch };
}
