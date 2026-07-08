"""Deterministic guards on the grounding prompts (no Ollama needed).

The prompts were rewritten to stop the small 3B model over-refusing broad
"what is this about" questions. These tests lock in the two invariants the rest
of the pipeline depends on: the LlamaIndex placeholders must survive any future
prompt edit, and the exact REFUSAL_MESSAGE must still be embedded (the refusal
short-circuit in engine.py/chat.py compares against it).
"""
from app.qa.prompts import (
    GROUNDED_CHAT_CONTEXT_PROMPT,
    GROUNDED_QA_PROMPT,
    REFUSAL_MESSAGE,
    _GROUNDED_QA_TMPL,
)


def test_qa_prompt_keeps_required_placeholders_and_refusal():
    """The single-shot QA template must expose {context_str}/{query_str} (LlamaIndex
    fills these) and embed the exact refusal string.
    """
    assert "{context_str}" in _GROUNDED_QA_TMPL
    assert "{query_str}" in _GROUNDED_QA_TMPL
    assert REFUSAL_MESSAGE in _GROUNDED_QA_TMPL
    # PromptTemplate must have parsed both variables out of the template.
    assert set(GROUNDED_QA_PROMPT.template_vars) == {"context_str", "query_str"}


def test_chat_context_prompt_keeps_context_placeholder_and_refusal():
    """The chat context prompt injects retrieved context via {context_str} and embeds
    the exact refusal string; it has no {query_str} (the condensed question is a
    separate user turn, not templated in).
    """
    assert "{context_str}" in GROUNDED_CHAT_CONTEXT_PROMPT
    assert "{query_str}" not in GROUNDED_CHAT_CONTEXT_PROMPT
    assert REFUSAL_MESSAGE in GROUNDED_CHAT_CONTEXT_PROMPT


def test_prompts_permit_summarization():
    """Regression guard for the reported bug: both prompts must explicitly allow
    summarizing (not just literal extraction), so broad questions aren't refused.
    """
    assert "summarize" in _GROUNDED_QA_TMPL.lower()
    assert "summarize" in GROUNDED_CHAT_CONTEXT_PROMPT.lower()
