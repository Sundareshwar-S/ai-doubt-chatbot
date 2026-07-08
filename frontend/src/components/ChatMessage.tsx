import type { ChatMessage as ChatMessageType } from '../hooks/useChat';
import { BotIcon, UserIcon } from './icons';

export function ChatMessage({ message }: { message: ChatMessageType }) {
  return (
    <div className={`chat-message-row chat-message-row--${message.role}`}>
      <span className={`avatar avatar--${message.role === 'assistant' ? 'bot' : 'user'}`}>
        {message.role === 'assistant' ? <BotIcon /> : <UserIcon />}
      </span>
      <div className={`chat-message chat-message--${message.role}`}>
        <p>{message.content}</p>
        {message.citations && message.citations.length > 0 && (
          <div className="citations">
            {message.citations.map((citation) => (
              <span key={`${citation.source}-${citation.page}`} className="citation-chip">
                {citation.source} p.{citation.page}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
