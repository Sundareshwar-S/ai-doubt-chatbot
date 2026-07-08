// Shared library state: a single instance is created once in App.tsx and
// passed down to UploadPanel and LibraryList as props, so an upload in one
// component and the listing in the other stay in sync (two independent
// useDocuments() calls would each own separate state).
import { useCallback, useEffect, useState } from 'react';
import { apiFetch, ApiError } from '../api/client';
import type { DocumentEntry, DocumentListResponse, DocumentUploadResponse } from '../api/types';

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [removingSources, setRemovingSources] = useState<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    try {
      const result = await apiFetch<DocumentListResponse>('/api/documents');
      setDocuments(result.documents);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not load the library.');
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const uploadDocument = useCallback(
    async (file: File): Promise<DocumentUploadResponse> => {
      setIsUploading(true);
      setError(null);
      try {
        const formData = new FormData();
        formData.append('file', file);
        const result = await apiFetch<DocumentUploadResponse>('/api/documents', {
          method: 'POST',
          body: formData,
        });
        await refresh();
        return result;
      } catch (err) {
        setError(err instanceof ApiError ? err.message : 'Upload failed.');
        throw err;
      } finally {
        setIsUploading(false);
      }
    },
    [refresh],
  );

  const removeDocument = useCallback(
    async (source: string) => {
      // Guards against a rapid double-click firing two DELETEs for the same
      // document: the second would 404 on an already-removed source and
      // surface a spurious error for what the user experienced as one click.
      if (removingSources.has(source)) {
        return;
      }
      setError(null);
      setRemovingSources((prev) => new Set(prev).add(source));
      try {
        await apiFetch<void>(`/api/documents/${encodeURIComponent(source)}`, {
          method: 'DELETE',
        });
        await refresh();
      } catch (err) {
        setError(err instanceof ApiError ? err.message : 'Could not remove the document.');
        throw err;
      } finally {
        setRemovingSources((prev) => {
          const next = new Set(prev);
          next.delete(source);
          return next;
        });
      }
    },
    [refresh, removingSources],
  );

  return {
    documents,
    error,
    isUploading,
    removingSources,
    uploadDocument,
    removeDocument,
    refresh,
  };
}

export type UseDocumentsResult = ReturnType<typeof useDocuments>;
