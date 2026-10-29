"""Unit tests for DeterministicGuardrail entity gating behavior."""
import pytest
from evaluation.guardrails import DeterministicGuardrail
from evaluation.models import RetrievedChunk, RetrievalResult


def make_result(content: str, max_dense_cosine: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunks=[RetrievedChunk(chunk_id="c1", content=content, score=1.0, rank=1)],
        diagnostics={"max_dense_cosine": max_dense_cosine},
    )


def test_guardrail_refuses_absent_entity():
    # Case (a): query naming an ID absent from the chunks -> refused
    guard = DeterministicGuardrail(tau_lex=0.0, tau_dense=0.0)
    result = make_result("Transaction TR101 was settled successfully.")
    decision = guard.evaluate("What happened to TR9999?", result)
    assert decision.refuse is True
    assert decision.reason.startswith("entity_absent")
    assert "TR9999" in decision.reason


@pytest.mark.parametrize("query, chunk_text", [
    ("Status of TR109 today", "Transaction tr109 settled successfully with all fees accounted."),
    ("Status of tr109 today", "Transaction TR109 settled successfully with all fees accounted."),
    ("Details on 2024-INV-001", "Invoice 2024-inv-001 was paid in full on schedule."),
    ("Details on 2024-inv-001", "Invoice 2024-INV-001 was paid in full on schedule."),
    ("Check item E-4017 now", "Part e-4017 verified in inventory database today."),
])
def test_guardrail_allows_present_entity_variants(query, chunk_text):
    # Case (b): ID present in a chunk (including lowercase and hyphenated variants) -> allowed
    guard = DeterministicGuardrail(tau_lex=0.1, tau_dense=0.0)
    result = make_result(chunk_text, max_dense_cosine=0.95)
    decision = guard.evaluate(query, result)
    assert not decision.reason.startswith("entity_absent")
    assert decision.refuse is False


def test_guardrail_ordinal_not_refused_for_entity():
    # Case (c): "What was revenue in the 4th quarter?" with chunks lacking token "4th"
    # -> NOT refused for entity reasons
    guard = DeterministicGuardrail(tau_lex=0.0, tau_dense=0.0)
    # The chunk describes revenue for the fourth quarter without using "4th"
    result = make_result("Quarterly revenue reached 30 billion dollars in the final period.")
    decision = guard.evaluate("What was revenue in the 4th quarter?", result)
    assert not decision.reason.startswith("entity_absent")
    assert decision.refuse is False


def test_guardrail_form_name_not_refused_for_entity():
    # Case (d): "What does the 10-K say?" likewise not refused for entity reasons
    guard = DeterministicGuardrail(tau_lex=0.0, tau_dense=0.0)
    result = make_result("The annual filing discusses primary market risks and supply chains.")
    decision = guard.evaluate("What does the 10-K say?", result)
    assert not decision.reason.startswith("entity_absent")
    assert decision.refuse is False

