import { BotIcon } from './icons';

export function TypingIndicator() {
  return (
    <div className="chat-message-row chat-message-row--assistant">
      <span className="avatar avatar--bot">
        <BotIcon />
      </span>
      <div className="chat-message chat-message--assistant chat-message--loading">
        <span className="spinner" role="status" aria-label="Generating answer" />
      </div>
    </div>
  );
}
