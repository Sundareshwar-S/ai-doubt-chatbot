"""Grounding prompt: answer only from retrieved context, cite source+page, refuse otherwise."""
from __future__ import annotations

from llama_index.core import PromptTemplate

REFUSAL_MESSAGE = "I can't find this in your documents."

_GROUNDED_QA_TMPL = (
    "Context information from the user's documents is below. Each chunk is preceded by its "
    "source file and page number.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using only the context above, and no prior knowledge, answer the question.\n"
    f'If the answer is not contained in the context, respond exactly with: "{REFUSAL_MESSAGE}"\n'
    "Otherwise, cite the source file and page number for every fact you use, "
    "e.g. (source.pdf, p.3).\n"
    "Question: {query_str}\n"
    "Answer: "
)

GROUNDED_QA_PROMPT = PromptTemplate(_GROUNDED_QA_TMPL)

# Context prompt for the multi-turn chat engine (Phase 4). CondensePlusContextChatEngine
# expects a plain string with a `{context_str}` placeholder (not a PromptTemplate); the
# condensed standalone question is appended as the user turn after this system context.
GROUNDED_CHAT_CONTEXT_PROMPT = (
    "You are answering questions about the user's own documents. Retrieved context is below. "
    "Each chunk is preceded by its source file and page number.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using only the context above and the conversation so far, and no prior knowledge, answer "
    "the user's question.\n"
    f'If the answer is not contained in the context, respond exactly with: "{REFUSAL_MESSAGE}"\n'
    "Otherwise, cite the source file and page number for every fact you use, "
    "e.g. (source.pdf, p.3)."
)
