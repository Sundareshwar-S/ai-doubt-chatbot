import { useHealth } from '../hooks/useHealth';

// D4: a stopped Ollama server (or a missing model) surfaces here as a
// readable banner instead of silent failures deeper in the upload/chat flow.
export function StatusBanner() {
  const { health, refetch } = useHealth();

  if (!health || health.status === 'ok') {
    return null;
  }

  return (
    <div className="status-banner" role="alert">
      <span>{health.detail ?? 'The backend is degraded.'}</span>
      <button type="button" onClick={() => refetch()}>
        Recheck
      </button>
    </div>
  );
}
