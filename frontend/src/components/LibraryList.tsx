import type { UseDocumentsResult } from '../hooks/useDocuments';

type LibraryListProps = Pick<UseDocumentsResult, 'documents' | 'removeDocument' | 'removingSources'>;

export function LibraryList({ documents, removeDocument, removingSources }: LibraryListProps) {
  return (
    <section className="library-list">
      <h2>Library</h2>
      {documents.length === 0 ? (
        <p className="status-text">No documents uploaded yet.</p>
      ) : (
        <ul>
          {documents.map((doc) => (
            <li key={doc.source}>
              <span>
                {doc.source} ({doc.pages} page{doc.pages === 1 ? '' : 's'})
              </span>
              <button
                type="button"
                aria-label={`Remove ${doc.source}`}
                disabled={removingSources.has(doc.source)}
                onClick={() => removeDocument(doc.source)}
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
