import { useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import type { UseDocumentsResult } from '../hooks/useDocuments';
import { CloudUploadIcon } from './icons';

type UploadPanelProps = Pick<UseDocumentsResult, 'uploadDocument' | 'isUploading'>;

export function UploadPanel({ uploadDocument, isUploading }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [lastMessage, setLastMessage] = useState<string | null>(null);

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setLastMessage(null);
    try {
      const result = await uploadDocument(file);
      setLastMessage(
        result.ingested
          ? `"${result.source}" added to the library.`
          : `"${result.source}" was already in the library.`,
      );
    } catch {
      // uploadDocument() already recorded the failure in `error`.
    } finally {
      if (inputRef.current) {
        inputRef.current.value = '';
      }
    }
  };

  return (
    <section className="upload-panel">
      <span className="icon-badge">
        <CloudUploadIcon />
      </span>
      <h2>Upload a document</h2>
      <label htmlFor="doc-upload" className="visually-hidden">
        Choose a PDF or image file to upload
      </label>
      <input
        id="doc-upload"
        ref={inputRef}
        type="file"
        accept=".pdf,.png,.jpg,.jpeg"
        onChange={handleFileChange}
        disabled={isUploading}
      />
      {isUploading && <p className="status-text">Uploading…</p>}
      {lastMessage && <p className="status-text">{lastMessage}</p>}
    </section>
  );
}
