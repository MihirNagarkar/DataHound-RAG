"""Regression tests for the shared record-ID extractor (entities.py)."""
import pytest

from entities import extract_entities


@pytest.mark.parametrize("query, expected", [
    ("Details on transaction TR109", ["TR109"]),
    ("Details on transaction tr109", ["tr109"]),
    ("Amount for INV-2023-001", ["INV-2023-001"]),
    ("What happened to TR9999 and E-9999?", ["TR9999", "E-9999"]),
    ("Show 2024-INV-001 and TR-2024", ["2024-INV-001", "TR-2024"]),
    ("Settlement CHN_77 in Q4 FY25", ["CHN_77"]),
    ("Look up PO/8841", []),  # slash IDs: documented gap
])
def test_real_ids_are_extracted(query, expected):
    assert extract_entities(query) == expected


@pytest.mark.parametrize("query", [
    "What was revenue in the 4th quarter?",
    "Summarize the 2nd half outlook",
    "What does the 10-K say about risk?",
    "See the 8-K and the 10-Q",
    "Compare 2024-Q4 vs 2023-Q4 revenue",
    "Data center revenue for Q4-FY25",
    "Revenue in FY2024 vs FY25",
    "Q4 FY25 gross margin was 73.0%",
    "Compare H100 and Blackwell B200",
    "How big is the 5G opportunity?",
    "What is the 401k plan?",
    "Plain sentence with no identifiers at all.",
    "",
])
def test_ordinary_phrases_are_not_ids(query):
    assert extract_entities(query) == []


@pytest.mark.parametrize("query, expected", [
    ("Sales of 1st-gen products", ["1st-gen"]),
    ("COVID-19 impact", ["COVID-19"]),
])
def test_known_false_positives_are_pinned(query, expected):
    assert extract_entities(query) == expected


def test_duplicates_and_order_preserved():
    assert extract_entities("TR1 then TR2 then TR1") == ["TR1", "TR2", "TR1"]


def test_pure_numbers_and_dates_are_not_ids():
    assert extract_entities("Transactions on 05-01-2016 totaling 3097.22 USD") == []


def test_all_layers_share_one_extractor():
    import entities
    from evaluation import guardrails
    assert guardrails.extract_entities is entities.extract_entities


@pytest.mark.parametrize("module_name", ["retriever", "app"])
def test_other_layers_share_one_extractor(module_name):
    import entities
    mod = pytest.importorskip(module_name)  # skipped if faiss/flask are absent
    assert mod.extract_entities is entities.extract_entities

