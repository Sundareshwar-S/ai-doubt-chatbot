import './App.css';
import { StatusBanner } from './components/StatusBanner';
import { UploadPanel } from './components/UploadPanel';
import { LibraryList } from './components/LibraryList';
import { ChatWindow } from './components/ChatWindow';
import { useDocuments } from './hooks/useDocuments';

function App() {
  // Created once here (not inside UploadPanel/LibraryList) so both
  // components share one document list instead of two independent copies.
  const documentsState = useDocuments();

  return (
    <div className="app">
      <header className="app-header">
        <h1>AI Doubt Solver</h1>
      </header>
      <StatusBanner />
      <main className="app-main">
        {documentsState.error && (
          <p className="error-text" role="alert">
            {documentsState.error}
          </p>
        )}
        <UploadPanel
          uploadDocument={documentsState.uploadDocument}
          isUploading={documentsState.isUploading}
        />
        <LibraryList
          documents={documentsState.documents}
          removeDocument={documentsState.removeDocument}
          removingSources={documentsState.removingSources}
        />
        <ChatWindow />
      </main>
    </div>
  );
}

export default App;
