"""Pandas aggregation, ASCII terminal dashboards, and GitHub-Flavored Markdown exporters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from .datasets import CATEGORIES, MOCK_CORPUS

MODE_ORDER = ["BM25", "FAISS", "Hybrid"]


def overall_table(df: pd.DataFrame, k: int) -> pd.DataFrame:
    ind = df[~df.expected_refusal]
    ood = df[df.expected_refusal]
    modes = [m for m in MODE_ORDER if m in set(df["mode"])]
    out = pd.DataFrame(index=modes)
    out[f"Hit Rate@{k}"] = ind.groupby("mode")["hit"].mean()
    out[f"MRR@{k}"] = ind.groupby("mode")["rr"].mean()
    out[f"Recall@{k}"] = ind.groupby("mode")["recall"].mean()
    out["Refusal Acc (OOD)"] = ood.groupby("mode")["refusal_correct"].mean()
    if "end_to_end_abstention" in ood.columns and ood["end_to_end_abstention"].notna().any():
        out["End-to-end abstention (gate OR model)"] = ood.groupby("mode")["end_to_end_abstention"].mean()
    out["False Refusal (in-domain)"] = ind.groupby("mode")["refused"].mean()
    for col, label in (("faithfulness", "Faithfulness"), ("answer_relevance", "Answer Relevance"),
                       ("context_precision", "Context Precision")):
        if df[col].notna().any():
            out[label] = ind.groupby("mode")[col].mean()
    out["Retrieval ms"] = df.groupby("mode")["retrieval_ms"].mean()
    if df["generation_ms"].notna().any():
        out["Generation ms"] = df.groupby("mode")["generation_ms"].mean()
    return out.loc[modes]


def category_matrix(df: pd.DataFrame, metric_col: str, ood: bool = False) -> pd.DataFrame:
    sub = df[df.category.isin(["out_of_domain_attack"])] if ood else df[~df.expected_refusal]
    cats = [c for c in CATEGORIES if c in set(sub.category)]
    m = sub.pivot_table(index="category", columns="mode", values=metric_col, aggfunc="mean")
    return m.reindex(index=cats, columns=[x for x in MODE_ORDER if x in m.columns])


def chunking_table(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    g = df.groupby(["strategy", "mode"])
    out = pd.DataFrame({f"Hit@{k}": g["hit"].mean(), f"MRR@{k}": g["rr"].mean()})
    if df["answer_correct"].notna().any():
        out["Answer Correct (top-1 ctx)"] = g["answer_correct"].mean()
    out["Retrieval ms"] = g["retrieval_ms"].mean()
    out = out.reset_index()
    out["strategy"] = pd.Categorical(out["strategy"], ["Naive", "Header-Pinned"])
    out["mode"] = pd.Categorical(out["mode"], MODE_ORDER)
    out = out.sort_values(["strategy", "mode"])
    out.index = [f"{s} / {m}" for s, m in zip(out["strategy"], out["mode"])]
    return out.drop(columns=["strategy", "mode"])


def _fmt(v: Any, digits: int = 3) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"


def render_ascii_table(title: str, table: pd.DataFrame, index_name: str = "") -> str:
    cols = [index_name] + [str(c) for c in table.columns]
    body = [[str(i)] + [_fmt(float(v)) for v in table.loc[i]] for i in table.index]
    widths = [max(len(r[j]) for r in [cols] + body) for j in range(len(cols))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    line = lambda r: "| " + " | ".join(c.ljust(w) if j == 0 else c.rjust(w)
                                       for j, (c, w) in enumerate(zip(r, widths))) + " |"
    head = f"{title:^{len(sep)}}"
    return "\n".join([head, sep, line(cols), sep] + [line(r) for r in body] + [sep])


def judge_coverage_note(df: pd.DataFrame) -> Optional[str]:
    """Report how many generated in-domain answers actually received judge scores."""
    gen = df[(~df.expected_refusal) & (~df.refused) & df.answer.notna() & (df.answer != "")]
    if gen.empty:
        return None
    ok = int(gen.faithfulness.notna().sum())
    return f"Judge scores parsed for {ok}/{len(gen)} generated in-domain answers."


def print_dashboard(df: pd.DataFrame, k: int, rrf_k: int, llm_model: Optional[str],
                    notes: Optional[Sequence[str]] = None, corpus_size: int = len(MOCK_CORPUS),
                    header_line: Optional[str] = None) -> None:
    bar = "=" * 78
    sub = f"k={k} | RRF k={rrf_k} | LLM={llm_model or 'n/a'} | queries={df.query_id.nunique()}"
    print(f"\n{bar}\n{'HYBRID RAG ABLATION BENCHMARK':^78}\n{sub:^78}")
    if header_line:
        print(f"{header_line:^78}")
    print(f"{bar}")
    print(render_ascii_table("[1] OVERALL MACRO-AVERAGES (in-domain unless noted)", overall_table(df, k).T.T,
                             "Mode"))
    for col, label in (("hit", f"Hit Rate@{k}"), ("rr", f"MRR@{k}"), ("recall", f"Recall@{k}")):
        print("\n" + render_ascii_table(f"[2] CATEGORY MATRIX - {label}", category_matrix(df, col), "Category"))
    print("\n" + render_ascii_table("[2] CATEGORY MATRIX - Refusal Accuracy (OOD)",
                                    category_matrix(df, "refusal_correct", ood=True), "Category"))
    for n in list(notes or []) + [judge_coverage_note(df) or ""]:
        if n:
            print(f"NOTE: {n}")
    print(bar)


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _md_table(
    table: pd.DataFrame,
    index_name: str,
    bold_best: Optional[str] = "max",
    bold_axis: str = "col",
) -> str:
    lines = ["| " + " | ".join([_escape_md_cell(index_name)] + [_escape_md_cell(str(c)) for c in table.columns]) + " |",
             "|" + "|".join(["---"] + ["---:"] * len(table.columns)) + "|"]
    col_best: Dict[Any, float] = {}
    if bold_best and bold_axis == "col":
        for c in table.columns:
            ser = table[c].dropna()
            if not ser.empty:
                col_best[c] = ser.min() if ("ms" in str(c) or "False" in str(c)) else ser.max()
    for idx, row in table.iterrows():
        row_ser = row.dropna()
        row_best = (row_ser.max() if bold_best == "max" else row_ser.min()) if (bold_best and not row_ser.empty) else None
        cells = []
        for c in table.columns:
            v = row[c]
            txt = _fmt(float(v), 1 if "ms" in str(c) else 3)
            if bold_best and v == v:
                if bold_axis == "col" and c in col_best and v == col_best[c] and table[c].nunique() > 1:
                    txt = f"**{txt}**"
                elif bold_axis == "row" and row_best is not None and v == row_best and row_ser.nunique() > 1:
                    txt = f"**{txt}**"
            cells.append(txt)
        lines.append("| " + " | ".join([_escape_md_cell(str(idx))] + cells) + " |")
    return "\n".join(lines)


def export_markdown(df: pd.DataFrame, path: Path, k: int, rrf_k: int, llm_model: Optional[str],
                    notes: Optional[Sequence[str]] = None, corpus_size: int = len(MOCK_CORPUS),
                    header_line: Optional[str] = None) -> None:
    parts = [
        "# Hybrid RAG Ablation Benchmark",
    ]
    if header_line:
        parts.append(f"\n> **Provenance:** `{header_line}`\n")
    parts.extend([
        f"\n*Config:* `k={k}`, `RRF k={rrf_k}`, generator/judge `{llm_model or 'n/a (generation skipped)'}`, "
        f"{df.query_id.nunique()} queries over a {corpus_size}-chunk corpus. "
        "IR metrics are macro-averaged over in-domain queries only; out-of-domain attacks are scored "
        "solely by Refusal Accuracy so they do not distort retrieval scores.\n",
        "## 1. Overall macro-averages (best per metric in **bold**)\n",
        _md_table(overall_table(df, k), "Mode", bold_axis="col"),
    ])
    for sub_id, (col, label) in zip(("2a", "2b", "2c"),
                                    (("hit", f"Hit Rate@{k}"), ("rr", f"MRR@{k}"), ("recall", f"Recall@{k}"))):
        parts += [f"\n## {sub_id}. Category matrix - {label} (best mode per row in **bold**)\n",
                  _md_table(category_matrix(df, col), "Category", bold_axis="row")]
    parts += ["\n## 3. Category matrix - Refusal Accuracy (out-of-domain)\n",
              _md_table(category_matrix(df, "refusal_correct", ood=True), "Category", bold_axis="row")]
    pq = df[["query_id", "category", "mode", "hit", "rr", "refused", "guardrail_reason", "retrieved_ids",
             "retrieval_ms"]].copy()
    pq["retrieval_ms"] = pq["retrieval_ms"].round(1)
    parts += ["\n## 4. Per-query detail\n",
              "| " + " | ".join(pq.columns) + " |", "|" + "|".join(["---"] * len(pq.columns)) + "|"]
    parts += ["| " + " | ".join("-" if (isinstance(v, float) and v != v) else str(v) for v in r) + " |"
              for r in pq.itertuples(index=False)]
    extra = list(notes or []) + [judge_coverage_note(df) or ""]
    if any(extra):
        parts += ["\n## Methodology notes\n"] + [f"- {n}" for n in extra if n]
        parts.append("- With only a handful of in-domain queries per category, differences are anecdotal, "
                     "not statistically significant.")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def report_chunking(df: pd.DataFrame, path: Path, llm_model: Optional[str],
                    header_line: Optional[str] = None) -> None:
    tbl = chunking_table(df)
    n = df.groupby("strategy")["n_chunks"].first().to_dict()
    print("\n" + render_ascii_table("CHUNKING ABLATION - Naive vs Header-Pinned (k=3, top-1 context)", tbl,
                                    "Strategy / Mode"))
    md = ["# Chunking Ablation: Naive vs Header-Pinned Table Chunks\n"]
    if header_line:
        md.append(f"> **Provenance:** `{header_line}`\n")
    q_count = df.query_id.nunique()
    md.extend([
          f"*Setup:* 60-row transaction ledger, {q_count} single-row lookup questions, chunks per strategy: {n}. "
          f"Relevance = the chunk contains the full gold row. Generation uses only the top-1 chunk as context "
          f"(`{llm_model or 'n/a, generation skipped'}`); correctness = gold cell value appears in the answer.\n",
          _md_table(tbl, "Strategy / Mode"),
          "\n- Fee and tax are both small decimals, so a chunk without column headers is ambiguous to the reader.",
          f"- {q_count} queries is a targeted sample."]
    )
    path.write_text("\n".join(md) + "\n", encoding="utf-8")

