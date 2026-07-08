// No session/conversation ID anywhere in here: the backend holds exactly one
// shared `chat_engine` (app/api/state.py), not one per browser tab -- see
// plan.md's "single shared conversation" decision for this personal,
// single-user, no-auth tool. Two tabs open at once share one conversation;
// "Clear chat" (resetChat) is the escape hatch.
import { useCallback, useRef, useState } from 'react';
import { ApiError } from '../api/client';
import type { ChatCitation, ChatStreamEvent, ErrorEnvelope } from '../api/types';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  citations?: ChatCitation[];
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Bumped by resetChat() so an in-flight askQuestion's streamed chunks, if
  // they arrive after a reset, are dropped instead of appending onto the
  // just-cleared transcript.
  const generationRef = useRef(0);

  // Concatenate a streamed token onto the trailing assistant message (the empty
  // placeholder appended when the turn started). No-op if the transcript was
  // reset out from under us.
  const appendToAssistant = useCallback((generation: number, text: string) => {
    if (generationRef.current !== generation) return;
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === 'assistant') {
        next[next.length - 1] = { ...last, content: last.content + text };
      }
      return next;
    });
  }, []);

  const setAssistantCitations = useCallback((generation: number, citations: ChatCitation[]) => {
    if (generationRef.current !== generation) return;
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === 'assistant') {
        next[next.length - 1] = { ...last, citations };
      }
      return next;
    });
  }, []);

  const askQuestion = useCallback(
    async (question: string) => {
      const generation = generationRef.current;
      setMessages((prev) => [
        ...prev,
        { role: 'user', content: question },
        { role: 'assistant', content: '' },
      ]);
      setIsAsking(true);
      setError(null);

      const fail = (message: string) => {
        if (generationRef.current !== generation) return;
        setError(message);
        // Drop the empty assistant placeholder if nothing streamed into it.
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.role === 'assistant' && last.content === '') {
            return prev.slice(0, -1);
          }
          return prev;
        });
      };

      try {
        let response: Response;
        try {
          response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
          });
        } catch {
          throw new ApiError(0, 'network_error', 'Could not reach the server.');
        }

        if (!response.ok || !response.body) {
          const body = (await response.json().catch(() => null)) as ErrorEnvelope | null;
          throw new ApiError(
            response.status,
            body?.error?.code ?? 'unknown_error',
            body?.error?.message ?? `Request failed with status ${response.status}.`,
          );
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        // Read NDJSON: each complete line is one ChatStreamEvent.
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let newline: number;
          while ((newline = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, newline).trim();
            buffer = buffer.slice(newline + 1);
            if (!line) continue;
            const event = JSON.parse(line) as ChatStreamEvent;
            if (event.type === 'token') {
              appendToAssistant(generation, event.text);
            } else if (event.type === 'done') {
              setAssistantCitations(generation, event.citations);
            } else if (event.type === 'error') {
              fail(event.message);
              return;
            }
          }
        }
      } catch (err) {
        fail(err instanceof ApiError ? err.message : 'Could not get an answer.');
      } finally {
        if (generationRef.current === generation) {
          setIsAsking(false);
        }
      }
    },
    [appendToAssistant, setAssistantCitations],
  );

  const resetChat = useCallback(async () => {
    try {
      // Mirror apiFetch's error handling without needing its generic return type.
      const response = await fetch('/api/chat/reset', { method: 'POST' });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as ErrorEnvelope | null;
        throw new ApiError(
          response.status,
          body?.error?.code ?? 'unknown_error',
          body?.error?.message ?? 'Could not clear the chat.',
        );
      }
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
