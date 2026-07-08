import { useState } from 'react';
import type { FormEvent } from 'react';
import { useChat } from '../hooks/useChat';
import { ChatMessage } from './ChatMessage';

export function ChatWindow() {
  const { messages, isAsking, error, askQuestion, resetChat } = useChat();
  const [question, setQuestion] = useState('');

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || isAsking) {
      return;
    }
    setQuestion('');
    await askQuestion(trimmed);
  };

  return (
    <section className="chat-window">
      <div className="chat-window-header">
        <h2>Chat</h2>
        <button type="button" onClick={() => resetChat()} disabled={isAsking}>
          Clear chat
        </button>
      </div>
      <div className="chat-transcript" aria-live="polite" aria-relevant="additions">
        {messages.map((message, index) => (
          <ChatMessage key={index} message={message} />
        ))}
      </div>
      {error && (
        <p className="error-text" role="alert">
          {error}
        </p>
      )}
      <form onSubmit={handleSubmit}>
        <label htmlFor="chat-question" className="visually-hidden">
          Question
        </label>
        <input
          id="chat-question"
          type="text"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask a question about your documents…"
          disabled={isAsking}
        />
        <button type="submit" disabled={isAsking || !question.trim()}>
          {isAsking ? 'Asking…' : 'Ask'}
        </button>
      </form>
    </section>
  );
}
