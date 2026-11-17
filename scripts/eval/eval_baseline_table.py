"""Compute the full baseline comparison table (R@5, R@10, R@20, nDCG@20, Coverage).

Methodology:

- **Labels**: GPT-5-mini judgements with bidirectional lookup. A pair (A,B)
  is looked up as both (A,B) and (B,A).
- **Positive pool**: For each query, the union of GPT-positive candidates
  across the **pool methods** (non-reranked retrievers) at depth **pool_depth**
  (default 50). Reranked methods and external baselines are NOT included in
  the pool — they are only evaluated against it.
- **Recall@K**: Fraction of the query's positive pool that appears in this
  method's top-K. Averaged over queries that have at least one positive.
- **nDCG@20**: Computed over labelled candidates only (unlabelled excluded).
  Uses graded relevance: related -> 0.5 + 0.5*confidence,
  unrelated -> 0.5 - 0.5*confidence.
- **Coverage**: Fraction of each method's top-20 candidates that have any
  GPT label (positive or negative). Methods that retrieve out-of-corpus
  papers (embedding baselines, S2 API) have lower coverage because many
  candidates were never submitted to GPT-5-mini for judging.

Usage:
    uv run python scripts/eval/eval_baseline_table.py 500q
    uv run python scripts/eval/eval_baseline_table.py 125q
    uv run python scripts/eval/eval_baseline_table.py 625q
    uv run python scripts/eval/eval_baseline_table.py 100q
"""

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from genret.processing.llm_judge import JudgedPair, normalise_judged_pair_dict
from genret.processing.reranking_compare import calc_ndcg, convert_gpt_relevance
from genret.utils import mean

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POOL_DEPTH = 50
"""Depth at which pool methods contribute to the positive set."""


@dataclass(frozen=True, kw_only=True)
class EvalConfig:
    """Configuration for one evaluation set."""

    label_paths: Sequence[Path]
    methods: Mapping[str, str | Sequence[str]]
    pool_methods: Sequence[str]
    """Methods whose top-{POOL_DEPTH} results define the positive set."""


class EvalSet(str, Enum):
    """Available evaluation test sets."""

    Q100 = "100q"
    """Original retrieval dev set."""
    Q500 = "500q"
    """Original partial retrieval test set."""
    Q125 = "125q"
    """Retrieval set with only new 2025/26 papers not in 100/500q."""
    Q625 = "625q"
    """Union of 500q and 125q."""
    Q158 = "158q"
    """1k-corpus test split (vmhk setup): 158 queries, MiniLM-L12 dense."""


# Non-reranked retrievers used to build the positive pool
POOL_500Q = [
    "Lyra (mean_pool, 4-facet)",
    "Lyra (MaxSim)",
    "Lyra (mean_pool)",
    "BM25 (title + facets)",
]

POOL_125Q = [
    "Lyra (mean_pool, 4-facet)",
    "Lyra (MaxSim)",
    "Lyra (mean_pool)",
    "BM25 (title + facets)",
]

POOL_100Q = [
    "Lyra (mean_pool, 4-facet)",
    "Lyra (mean_pool)",
    "Lyra (MaxSim)",
]

# 1k-corpus pool: in-corpus non-reranked retrievers only (out-of-corpus
# S2/SPECTER2 are scored against the pool, not part of it).
POOL_158Q = [
    "Graph retriever",
    "MiniLM (title+abstract)",
    "BM25 (title + facets)",
    "Dense-only (facet)",
]


EVAL_SETS: dict[str, EvalConfig] = {
    "100q": EvalConfig(
        label_paths=[
            Path("output/reranking_eval/gpt5mini_judgements.jsonl"),
            Path("output/cross_facet_eval/gpt5mini_new_judgements.jsonl"),
        ],
        methods={
            "Lyra (mean_pool, 4-facet)": "output/reranking_eval/retrieval_top100.jsonl",
            "Lyra (mean_pool)": "output/cross_facet_eval/retrieval_top100.jsonl",
            "Lyra (MaxSim)": "output/maxsim_minilm/retrieval_100q.jsonl",
        },
        pool_methods=POOL_100Q,
    ),
    "500q": EvalConfig(
        label_paths=[
            Path("data/s2orc-nlp/splits-connected-10k/processed/llm_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_baseline_missing_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_missing_embedding_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_all_missing_judgements.jsonl"),
            Path("output/embedding_baselines_500q/gpt5mini_judgements.jsonl"),
        ],
        methods={
            "MiniLM (title+abstract)": "output/embedding_baselines_500q/minilm.jsonl",
            "SPECTER2 (title+abstract)": "output/embedding_baselines_500q/specter2.jsonl",
            "S2 Recommendations API": "output/s2_recs_baseline/s2_recs_500.jsonl",
            "BM25 (title + facets)": "output/bm25_baseline/retrieval_bm25.jsonl",
            "BM25 + ModernBERT rerank": "output/bm25_baseline/retrieval_bm25_reranked.jsonl",
            "Lyra (mean_pool, 4-facet)": "output/eval_500/retrieval_baseline.jsonl",
            "Lyra (mean_pool) + rerank": "output/eval_500/retrieval_cross_facet_reranked.jsonl",
            "Lyra (MaxSim) + rerank": "output/eval_500/retrieval_minilm12_reranked.jsonl",
            "Lyra (MaxSim)": "output/eval_500/retrieval_minilm12.jsonl",
            "Lyra (mean_pool)": "output/eval_500/retrieval_cross_facet.jsonl",
        },
        pool_methods=POOL_500Q,
    ),
    "125q": EvalConfig(
        label_paths=[
            Path("output/eval_v2/gpt5mini_judgements_125q.jsonl"),
        ],
        methods={
            "MiniLM (title+abstract)": "output/eval_v2/retrieval_minilm_emb_125q.jsonl",
            "SPECTER2 (title+abstract)": "output/eval_v2/retrieval_specter2_emb_125q.jsonl",
            "S2 Recommendations API": "output/eval_v2/retrieval_s2recs_125q.jsonl",
            "BM25 (title + facets)": "output/eval_v2/retrieval_bm25_125q.jsonl",
            "BM25 + ModernBERT rerank": "output/eval_v2/retrieval_bm25_reranked_125q.jsonl",
            "Lyra (mean_pool, 4-facet)": "output/eval_v2/retrieval_baseline_125q.jsonl",
            "Lyra (mean_pool) + rerank": "output/eval_v2/retrieval_crossfacet_reranked_125q.jsonl",
            "Lyra (MaxSim) + rerank": "output/eval_v2/retrieval_minilm12_reranked_125q.jsonl",
            "Lyra (MaxSim)": "output/eval_v2/retrieval_minilm12_125q.jsonl",
            "Lyra (mean_pool)": "output/eval_v2/retrieval_crossfacet_125q.jsonl",
        },
        pool_methods=POOL_125Q,
    ),
    "625q": EvalConfig(
        label_paths=[
            Path("data/s2orc-nlp/splits-connected-10k/processed/llm_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_baseline_missing_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_missing_embedding_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_all_missing_judgements.jsonl"),
            Path("output/embedding_baselines_500q/gpt5mini_judgements.jsonl"),
            Path("output/eval_v2/gpt5mini_judgements_125q.jsonl"),
        ],
        methods={
            "MiniLM (title+abstract)": [
                "output/embedding_baselines_500q/minilm.jsonl",
                "output/eval_v2/retrieval_minilm_emb_125q.jsonl",
            ],
            "SPECTER2 (title+abstract)": [
                "output/embedding_baselines_500q/specter2.jsonl",
                "output/eval_v2/retrieval_specter2_emb_125q.jsonl",
            ],
            "S2 Recommendations API": [
                "output/s2_recs_baseline/s2_recs_500.jsonl",
                "output/eval_v2/retrieval_s2recs_125q.jsonl",
            ],
            "BM25 (title + facets)": [
                "output/bm25_baseline/retrieval_bm25.jsonl",
                "output/eval_v2/retrieval_bm25_125q.jsonl",
            ],
            "BM25 + ModernBERT rerank": [
                "output/bm25_baseline/retrieval_bm25_reranked.jsonl",
                "output/eval_v2/retrieval_bm25_reranked_125q.jsonl",
            ],
            "Lyra (mean_pool, 4-facet)": [
                "output/eval_500/retrieval_baseline.jsonl",
                "output/eval_v2/retrieval_baseline_125q.jsonl",
            ],
            "Lyra (mean_pool) + rerank": [
                "output/eval_500/retrieval_cross_facet_reranked.jsonl",
                "output/eval_v2/retrieval_crossfacet_reranked_125q.jsonl",
            ],
            "Lyra (MaxSim) + rerank": [
                "output/eval_500/retrieval_minilm12_reranked.jsonl",
                "output/eval_v2/retrieval_minilm12_reranked_125q.jsonl",
            ],
            "Lyra (MaxSim)": [
                "output/eval_500/retrieval_minilm12.jsonl",
                "output/eval_v2/retrieval_minilm12_125q.jsonl",
            ],
            "Lyra (mean_pool)": [
                "output/eval_500/retrieval_cross_facet.jsonl",
                "output/eval_v2/retrieval_crossfacet_125q.jsonl",
            ],
        },
        pool_methods=POOL_500Q,  # same pool methods as 500q
    ),
    "158q": EvalConfig(
        label_paths=[
            Path("output/eval_1k/gpt5mini_judgements_1k.jsonl"),
            Path("output/eval_500/gpt5mini_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_all_missing_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_baseline_missing_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_missing_embedding_judgements.jsonl"),
            Path("output/embedding_baselines_500q/gpt5mini_judgements.jsonl"),
        ],
        methods={
            "MiniLM (title+abstract)": "output/eval_1k/retrieval_minilm.jsonl",
            "SPECTER2 (title+abstract)": "output/eval_1k/retrieval_specter2.jsonl",
            "S2 Recommendations API": "output/eval_1k/retrieval_s2recs.jsonl",
            "BM25 (title + facets)": "output/eval_1k/retrieval_bm25.jsonl",
            "BM25 + rerank": (
                "output/eval_1k/rerank_modernbert150k/retrieval_bm25__reranked.jsonl"
            ),
            "Dense-only (facet)": "output/eval_1k/retrieval_denseonly.jsonl",
            "Graph retriever": "output/eval_1k/retrieval_graph.jsonl",
            "Graph + rerank": (
                "output/eval_1k/rerank_modernbert150k/retrieval_graph__reranked.jsonl"
            ),
            "Graph + Qwen3-32B": (
                "output/eval_1k/rerank_qwen/retrieval_graph__reranked.jsonl"
            ),
            "Graph + GTE-150k": (
                "output/eval_1k/rerank_gte150k/retrieval_graph__reranked.jsonl"
            ),
            "Graph + ModernBERT-55k": (
                "output/eval_1k/rerank_modernbert55k/retrieval_graph__reranked.jsonl"
            ),
        },
        pool_methods=POOL_158Q,
    ),
}


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


def load_labels(paths: Iterable[Path]) -> dict[tuple[int, int], JudgedPair]:
    """Load GPT labels with bidirectional keys."""
    labels: dict[tuple[int, int], JudgedPair] = {}
    for p in paths:
        if not p.exists():
            logger.warning("Label file not found: %s", p)
            continue

        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue

                with suppress(Exception):
                    r: dict[str, Any] = json.loads(line)
                    pair = JudgedPair.model_validate(normalise_judged_pair_dict(r))
                    labels[(pair.paper_a, pair.paper_b)] = pair
                    labels[(pair.paper_b, pair.paper_a)] = pair
    return labels


# ---------------------------------------------------------------------------
# Retrieval loading
# ---------------------------------------------------------------------------


def load_retrieval(paths: str | Sequence[str]) -> dict[int, list[int]]:
    """Load retrieval results. Supports single path or list (merged)."""
    if isinstance(paths, str):
        paths = [paths]

    results: dict[int, list[int]] = {}
    for p in map(Path, paths):
        if not p.exists():
            logger.warning("Retrieval file not found: %s", p)
            continue

        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue

                r: dict[str, Any] = json.loads(line)
                q = int(r["corpusid"])
                if q not in results:
                    results[q] = [res["corpusid"] for res in r.get("results", [])[:50]]
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodResult:
    """Evaluation metrics for a single retrieval method."""

    recall_5: float
    recall_10: float
    recall_20: float
    ndcg_20: float
    coverage: float
    """Fraction of top-20 candidates with any GPT label (0-100)."""
    queries: int
    """Number of queries with at least one positive in the pool."""


@dataclass
class _MethodAccumulator:
    """Mutable accumulator used during metric computation."""

    recalls: dict[int, list[float]] = field(
        default_factory=lambda: {5: [], 10: [], 20: []}
    )
    ndcgs: list[float] = field(default_factory=list[float])
    total_queries: int = 0
    labelled_queries: int = 0
    total_candidates: int = 0
    labelled_candidates: int = 0


@dataclass(frozen=True)
class TableResult:
    """Output of compute_table."""

    methods: Mapping[str, MethodResult]
    n_queries_with_positives: int
    n_total_positives: int
    avg_positives_per_query: float


@dataclass(frozen=True)
class HtmlSection:
    """One table section for HTML rendering."""

    title: str
    caption: str
    methods: Mapping[str, MethodResult]


def compute_table(
    cfg: EvalConfig,
    labels: dict[tuple[int, int], JudgedPair],
    top_k: int = 20,
) -> TableResult:
    """Compute R@5, R@10, R@20, nDCG@20, Coverage for each method.

    1. Load all retrievals.
    2. Build per-query positive set from pool methods at POOL_DEPTH.
    3. For each method, compute Recall against the pool and nDCG over
       labelled candidates only.
    """
    all_retrievals: dict[str, dict[int, list[int]]] = {}
    for name, paths in cfg.methods.items():
        all_retrievals[name] = load_retrieval(paths)

    # Build positive set from pool methods only, at POOL_DEPTH
    query_positives: dict[int, set[int]] = defaultdict(set)
    for pool_name in cfg.pool_methods:
        retrieval = all_retrievals.get(pool_name, {})
        for q, candidates in retrieval.items():
            for c in candidates[:POOL_DEPTH]:
                if (q, c) in labels and labels[(q, c)].related:
                    query_positives[q].add(c)

    n_with_pos = len(query_positives)
    n_pos = sum(len(v) for v in query_positives.values())
    console.print(f"Pool methods: {cfg.pool_methods}")
    console.print(f"Pool depth: {POOL_DEPTH}")
    console.print(f"Queries with positives: {n_with_pos}")
    console.print(f"Total positives: {n_pos}")
    console.print(f"Avg positives/query: {n_pos / max(n_with_pos, 1):.1f}")

    # Compute metrics per method
    results: dict[str, MethodResult] = {}
    for name, retrieval in all_retrievals.items():
        acc = _MethodAccumulator(total_queries=len(retrieval))

        for q, candidates in retrieval.items():
            top = candidates[:top_k]

            # Coverage: count labelled candidates in top-K
            for c in top:
                acc.total_candidates += 1
                if (q, c) in labels:
                    acc.labelled_candidates += 1

            positives = query_positives.get(q)
            if not positives:
                continue

            acc.labelled_queries += 1

            for k in [5, 10, 20]:
                found = sum(c in positives for c in top[:k])
                acc.recalls[k].append(found / len(positives))

            if scored := [
                (c, convert_gpt_relevance(labels[(q, c)]))
                for c in top
                if (q, c) in labels
            ]:
                rels = [rel for _, rel in scored]
                acc.ndcgs.append(calc_ndcg(rels, min(top_k, len(rels))))

        cov = (
            acc.labelled_candidates / acc.total_candidates * 100
            if acc.total_candidates
            else 0
        )
        results[name] = MethodResult(
            recall_5=mean(acc.recalls[5]),
            recall_10=mean(acc.recalls[10]),
            recall_20=mean(acc.recalls[20]),
            ndcg_20=mean(acc.ndcgs),
            coverage=cov,
            queries=acc.labelled_queries,
        )

    avg_pos = n_pos / max(n_with_pos, 1)
    return TableResult(
        methods=results,
        n_queries_with_positives=n_with_pos,
        n_total_positives=n_pos,
        avg_positives_per_query=avg_pos,
    )


METRIC_COLUMNS: list[tuple[str, str]] = [
    ("R@5", "recall_5"),
    ("R@10", "recall_10"),
    ("R@20", "recall_20"),
    ("nDCG@20", "ndcg_20"),
]
"""(display_name, field_name) pairs for metric columns."""

# Method grouping: methods within a group are separated from the next by a line
METHOD_GROUPS = [
    {"MiniLM (title+abstract)", "SPECTER2 (title+abstract)", "S2 Recommendations API"},
    {"BM25 (title + facets)", "BM25 + ModernBERT rerank"},
]


def render_table(title: str, results: Mapping[str, MethodResult]) -> None:
    """Print a Rich table of comparison results."""
    table = Table(title=title, show_lines=True)
    table.add_column("Method", style="bold")
    for display, _ in METRIC_COLUMNS:
        table.add_column(display, justify="right")
    table.add_column("Cov", justify="right")
    table.add_column("Q", justify="right")

    for name, m in results.items():
        table.add_row(
            name,
            *(f"{getattr(m, attr):.3f}" for _, attr in METRIC_COLUMNS),
            f"{m.coverage:.1f}%",
            str(m.queries),
        )

    console.print(table)


def _best_and_second(
    results: Mapping[str, MethodResult],
) -> dict[str, tuple[str, str]]:
    """For each metric field, find the method names with best and second-best values."""
    ranking: dict[str, tuple[str, str]] = {}
    for _, attr in METRIC_COLUMNS:
        sorted_methods = sorted(
            results.items(), key=lambda kv: getattr(kv[1], attr), reverse=True
        )
        best = sorted_methods[0][0] if len(sorted_methods) > 0 else ""
        second = sorted_methods[1][0] if len(sorted_methods) > 1 else ""
        ranking[attr] = (best, second)
    return ranking


def _is_group_boundary(name: str, prev_name: str | None) -> bool:
    """Check if there should be a group separator before this method."""
    if prev_name is None:
        return False
    for group in METHOD_GROUPS:
        if prev_name in group and name not in group:
            return True
        if name in group and prev_name not in group:
            return True
    return False


def render_html(sections: Sequence[HtmlSection]) -> str:
    """Render multiple result tables as a standalone HTML page."""
    style = """\
  body {
    font-family: "Times New Roman", Times, serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
    color: #222;
  }
  h1 { font-size: 1.4em; text-align: center; margin-bottom: 0.3em; }
  .subtitle { text-align: center; font-size: 0.95em; color: #555; margin-bottom: 2em; }
  h2 { font-size: 1.1em; margin-top: 2em; margin-bottom: 0.3em; }
  .table-caption { font-size: 0.9em; color: #444; margin-bottom: 0.5em; }
  table { border-collapse: collapse; width: 100%; font-size: 0.92em; margin-bottom: 2em; }
  thead th {
    border-top: 2px solid #222; border-bottom: 1px solid #222;
    padding: 6px 10px; text-align: right; font-weight: normal; font-style: italic;
  }
  thead th:first-child { text-align: left; font-style: normal; font-weight: bold; }
  tbody td { padding: 4px 10px; text-align: right; border: none; }
  tbody td:first-child { text-align: left; }
  tbody tr:last-child td { border-bottom: 2px solid #222; }
  tbody tr:hover { background: #f8f8f8; }
  .best { font-weight: bold; }
  .second { text-decoration: underline; text-underline-offset: 2px; }
  .group-sep td { border-top: 1px solid #ccc; }"""

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        "<title>Lyra Baseline Comparison</title>",
        f"<style>{style}</style>",
        "</head>",
        "<body>",
        "<h1>Lyra Retrieval Baseline Comparison</h1>",
        '<div class="subtitle">',
        "GPT-5-mini oracle &middot; "
        "Positive pool from non-reranked retrievers' top-50 &middot; "
        "Coverage = fraction of top-20 candidates with GPT labels",
        "</div>",
    ]

    cols = [display for display, _ in METRIC_COLUMNS] + ["Cov"]

    for section in sections:
        ranking = _best_and_second(section.methods)
        parts.append(f"<h2>{section.title}</h2>")
        parts.append(f'<div class="table-caption">{section.caption}</div>')
        parts.append("<table>")
        parts.append(
            "<thead><tr><th>Method</th>"
            + "".join(f"<th>{c}</th>" for c in cols)
            + "</tr></thead>"
        )
        parts.append("<tbody>")

        prev_name: str | None = None
        for name, m in section.methods.items():
            cls = ' class="group-sep"' if _is_group_boundary(name, prev_name) else ""
            parts.append(f"<tr{cls}>")
            parts.append(f"  <td>{name}</td>")
            for _, attr in METRIC_COLUMNS:
                val = getattr(m, attr)
                best, second = ranking[attr]
                if name == best:
                    parts.append(f'  <td class="best">{val:.3f}</td>')
                elif name == second:
                    parts.append(f'  <td class="second">{val:.3f}</td>')
                else:
                    parts.append(f"  <td>{val:.3f}</td>")
            parts.append(f"  <td>{m.coverage:.1f}%</td>")
            parts.append("</tr>")
            prev_name = name

        parts.append("</tbody></table>")

    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(no_args_is_help=True, help=__doc__)
def main(
    eval_sets: Annotated[
        list[EvalSet],
        typer.Argument(help="Evaluation set(s) to compute."),
    ],
    top_k: Annotated[
        int, typer.Option("--top-k", help="Top-K for recall and nDCG.")
    ] = 20,
    html: Annotated[
        Path | None,
        typer.Option("--html", help="Write HTML comparison file to this path."),
    ] = None,
) -> None:
    """Compute the baseline comparison table for one or more eval sets."""
    html_sections: list[HtmlSection] = []

    for eval_set in eval_sets:
        cfg = EVAL_SETS[eval_set.value]

        console.print(f"Loading labels for {eval_set.value}...")
        labels = load_labels(cfg.label_paths)
        console.print(f"Labels loaded: {len(labels):,} (bidirectional)\n")

        result = compute_table(cfg, labels, top_k=top_k)
        title = f"{eval_set.value} (top-{top_k})"
        render_table(title, result.methods)

        n_labels = len(labels) // 2
        html_sections.append(
            HtmlSection(
                title=title,
                caption=(
                    f"{n_labels:,} unique labels, "
                    f"avg {result.avg_positives_per_query:.1f} positives/query."
                ),
                methods=result.methods,
            )
        )

    if html is not None:
        html_content = render_html(html_sections)
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text(html_content)
        console.print(f"\nHTML written to {html}")


if __name__ == "__main__":
    app()
