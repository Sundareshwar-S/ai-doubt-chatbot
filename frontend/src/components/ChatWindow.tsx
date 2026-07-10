import { useState } from 'react';
import type { FormEvent } from 'react';
import { useChat } from '../hooks/useChat';
import { ChatMessage } from './ChatMessage';
import { TypingIndicator } from './TypingIndicator';
import { ThemeToggle } from './ThemeToggle';
import { SendIcon } from './icons';

export function ChatWindow() {
  const { messages, isAsking, error, askQuestion, resetChat } = useChat();
  const [question, setQuestion] = useState('');

  const lastMessage = messages[messages.length - 1];
  const showTyping = isAsking && lastMessage?.role === 'assistant' && lastMessage.content === '';
  const visibleMessages = showTyping ? messages.slice(0, -1) : messages;

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
      <div className="chat-header">
        <h2>Chat</h2>
        <div className="chat-header-actions">
          <ThemeToggle />
          <button
            type="button"
            className="clear-chat-button"
            onClick={() => resetChat()}
            disabled={isAsking}
          >
            Clear chat
          </button>
        </div>
      </div>
      <div className="chat-transcript" aria-live="polite" aria-relevant="additions">
        {visibleMessages.map((message, index) => (
          <ChatMessage key={index} message={message} />
        ))}
        {showTyping && <TypingIndicator />}
      </div>
      {error && (
        <p className="error-text" role="alert">
          {error}
        </p>
      )}
      <form className="chat-input-bar" onSubmit={handleSubmit}>
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
        <button
          type="submit"
          className="send-button"
          aria-label="Send question"
          disabled={isAsking || !question.trim()}
        >
          {isAsking ? (
            <span className="spinner" role="status" aria-label="Sending" />
          ) : (
            <SendIcon />
          )}
        </button>
      </form>
    </section>
  );
}
