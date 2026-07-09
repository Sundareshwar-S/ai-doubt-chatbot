"""Grounding prompt: answer only from retrieved context, cite source+page, refuse otherwise."""
from __future__ import annotations

from llama_index.core import PromptTemplate

REFUSAL_MESSAGE = "I can't find this in your documents."

_GROUNDED_QA_TMPL = (
    "The text below is the actual content of the user's own documents. Each chunk is preceded "
    "by its source file and page number.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Answer the question using the content above. You may summarize, explain, paraphrase, or "
    "combine facts across chunks — broad questions like \"what is this document about\" should be "
    "answered by summarizing the content above. Base your answer only on this content; do not use "
    "outside knowledge or invent facts it does not support.\n"
    "Only if the content above is unrelated to the question and gives you nothing to work with, "
    f'respond with exactly: "{REFUSAL_MESSAGE}"\n'
    "When you state a specific fact, cite its source file and page number, e.g. (source.pdf, p.3). "
    "Write any math using LaTeX, delimited with $...$ or \\(...\\) for inline expressions and "
    "$$...$$ or \\[...\\] for display equations (not square brackets or parentheses on their own). "
    "Keep the answer concise.\n"
    "Question: {query_str}\n"
    "Answer: "
)

GROUNDED_QA_PROMPT = PromptTemplate(_GROUNDED_QA_TMPL)

# Context prompt for the multi-turn chat engine (Phase 4). CondensePlusContextChatEngine
# expects a plain string with a `{context_str}` placeholder (not a PromptTemplate); the
# condensed standalone question is appended as the user turn after this system context.
GROUNDED_CHAT_CONTEXT_PROMPT = (
    "You are answering questions about the user's own documents. The text below is the actual "
    "content retrieved from those documents; each chunk is preceded by its source file and page "
    "number.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Answer using the content above together with the conversation so far. You may summarize, "
    "explain, paraphrase, or combine facts across chunks — broad questions like \"what is this "
    "document about\" should be answered by summarizing the content above. Base your answer only "
    "on this content and the conversation; do not use outside knowledge or invent facts it does "
    "not support.\n"
    "Only if the content above is unrelated to the question and gives you nothing to work with, "
    f'respond with exactly: "{REFUSAL_MESSAGE}"\n'
    "When you state a specific fact, cite its source file and page number, e.g. (source.pdf, p.3). "
    "Write any math using LaTeX, delimited with $...$ or \\(...\\) for inline expressions and "
    "$$...$$ or \\[...\\] for display equations (not square brackets or parentheses on their own). "
    "Keep the answer concise."
)
