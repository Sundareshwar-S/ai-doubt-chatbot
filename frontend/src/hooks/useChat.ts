// No session/conversation ID anywhere in here: the backend holds exactly one
// shared `chat_engine` (app/api/state.py), not one per browser tab -- see
// plan.md's "single shared conversation" decision for this personal,
// single-user, no-auth tool. Two tabs open at once share one conversation;
// "Clear chat" (resetChat) is the escape hatch.
import { useCallback, useRef, useState } from 'react';
import { apiFetch, ApiError } from '../api/client';
import type { ChatAskResponse, ChatCitation } from '../api/types';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  citations?: ChatCitation[];
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Bumped by resetChat() so an in-flight askQuestion's response, if it
  // resolves after a reset, is dropped instead of appending an orphaned
  // assistant reply onto the just-cleared transcript.
  const generationRef = useRef(0);

  const askQuestion = useCallback(async (question: string) => {
    const generation = generationRef.current;
    setMessages((prev) => [...prev, { role: 'user', content: question }]);
    setIsAsking(true);
    setError(null);
    try {
      const result = await apiFetch<ChatAskResponse>('/api/chat/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      if (generationRef.current !== generation) {
        return;
      }
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: result.answer, citations: result.citations },
      ]);
    } catch (err) {
      if (generationRef.current !== generation) {
        return;
      }
      setError(err instanceof ApiError ? err.message : 'Could not get an answer.');
    } finally {
      if (generationRef.current === generation) {
        setIsAsking(false);
      }
    }
  }, []);

  const resetChat = useCallback(async () => {
    try {
      await apiFetch<void>('/api/chat/reset', { method: 'POST' });
      generationRef.current += 1;
      setMessages([]);
      setError(null);
      setIsAsking(false);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not clear the chat.');
    }
  }, []);

  return { messages, isAsking, error, askQuestion, resetChat };
}
