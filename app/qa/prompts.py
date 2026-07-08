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
