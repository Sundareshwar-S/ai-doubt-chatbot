import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import type { ChatMessage as ChatMessageType } from '../hooks/useChat';
import { BotIcon, UserIcon } from './icons';

// The LLM reliably favors LaTeX's own \[...\]/\(...\) math delimiters over the
// $...$/$$...$$ the prompt asks for (a strong prior from its training data that
// prompting doesn't override) -- remark-math only recognizes $ delimiters, so
// translate before rendering instead of fighting the model's natural style.
function normalizeLatexDelimiters(text: string): string {
  return text
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, expr: string) => `$$${expr}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, expr: string) => `$${expr}$`);
}

export function ChatMessage({ message }: { message: ChatMessageType }) {
  return (
    <div className={`chat-message-row chat-message-row--${message.role}`}>
      <span className={`avatar avatar--${message.role === 'assistant' ? 'bot' : 'user'}`}>
        {message.role === 'assistant' ? <BotIcon /> : <UserIcon />}
      </span>
      <div className={`chat-message chat-message--${message.role}`}>
        <div className="chat-message-content">
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
            {normalizeLatexDelimiters(message.content)}
          </ReactMarkdown>
        </div>
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
