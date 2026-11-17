"""Score the ColBERT rebuttal baselines against GRAFT and the existing baselines.

Four gold settings on the 1k corpus:

- in-corpus, strict gold (Table 1 of the submission; 158 test queries). Despite
  the paper's wording it is citation-derived, not the graph neighbourhood: 88%
  direct references, the rest cited-by / bib-coupling / co-citation links.
- in-corpus, direct-citation gold (the query's references only; 158 queries)
- OOC unmatched, citation gold (404 queries, 351 with gold; Table 2)
- OOC k-core matched, citation gold (139 queries)

For each system it reports recall@{5,10,20,50} raw and with the popularity head
removed (the N most-cited corpus papers by in-degree over the NLP collection's
reference lists, dropped from gold and from every system's ranking, as in
docs/reference/20260801-ooc-exploratory-behaviour.md), plus a paired bootstrap
on per-query R@20 against GRAFT. Recall is averaged per query over queries with
non-empty gold, matching `score_glen_eval.py`.

Usage:
    uv run python scripts/eval/score_colbert_rebuttal.py \
        --colbert zeroshot=output/colbert_1k/retrieval_colbertv2_zs \
        --colbert replicated=output/colbert_1k/retrieval/replicated/checkpoint-3000

Each `--colbert NAME=PREFIX` expects `PREFIX_{incorpus,ooc,kcore}.jsonl`.
"""

import argparse
import importlib.util
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

KS = (5, 10, 20, 50)
"""Recall cutoffs."""
BOOTSTRAP_SAMPLES = 10_000
"""Paired bootstrap resamples."""


def _load_sibling(name: str) -> ModuleType:
    """Import a sibling script in scripts/eval as a module."""
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).parent / f"{name}.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sge = _load_sibling("score_glen_eval")
"""score_glen_eval.py, for gold loading and GRAFT fusion re-scoring."""

type Ranked = dict[int, list[int]]


@dataclass(frozen=True, kw_only=True)
class Setting:
    """One gold setting: a gold file, a query set and the systems to score."""

    name: str
    gold: Path
    queries: Path
    systems: Mapping[str, Ranked]
    reference: str
    """System the bootstrap compares every other system against."""


def load_ranked(path: Path) -> Ranked:
    """Load a retrieval JSONL as {query: [corpusid, ...]} in rank order."""
    return sge.load_eval(path)


def read_ids(path: Path) -> set[int]:
    """Read whitespace-separated corpus IDs."""
    return {int(x) for x in path.read_text().split()}


def popularity_head(references: Path, corpus: Set[int], size: int) -> frozenset[int]:
    """Return the `size` corpus papers with the highest citation in-degree."""
    degree: Counter[int] = Counter()
    with references.open() as f:
        for line in f:
            if line.strip():
                for target in json.loads(line)["references"]:
                    if int(target) in corpus:
                        degree[int(target)] += 1
    return frozenset(cid for cid, _ in degree.most_common(size))


def per_query_recall(
    ranked: Ranked,
    gold: Mapping[int, Set[int]],
    queries: Sequence[int],
    k: int,
    removed: Set[int],
) -> list[float]:
    """Recall@k per query, with `removed` papers dropped from gold and ranking."""
    values: list[float] = []
    for qid in queries:
        relevant = gold[qid] - removed
        top = [p for p in ranked.get(qid, []) if p not in removed][:k]
        values.append(sum(p in relevant for p in top) / len(relevant))
    return values


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], rng: random.Random
) -> tuple[float, float, float]:
    """Mean difference a - b with a 95% percentile paired-bootstrap CI."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return (
        sum(diffs) / n,
        means[int(0.025 * BOOTSTRAP_SAMPLES)],
        means[int(0.975 * BOOTSTRAP_SAMPLES)],
    )


def score_setting(setting: Setting, head: frozenset[int]) -> None:
    """Print raw and head-removed recall tables plus bootstrap vs the reference."""
    gold = sge.load_strict_gold(setting.gold)
    query_set = read_ids(setting.queries)
    for label, removed in (("raw", frozenset[int]()), ("head-removed", head)):
        queries = sorted(q for q in query_set if gold.get(q, set()) - removed)
        print(f"\n### {setting.name} -- {label} (n={len(queries)})\n")
        print(
            "| System | "
            + " | ".join(f"R@{k}" for k in KS)
            + " | ΔR@20 vs ref [95% CI] |"
        )
        print("|---|" + "---|" * (len(KS) + 1))
        ref_r20 = per_query_recall(
            setting.systems[setting.reference], gold, queries, 20, removed
        )
        for name, ranked in setting.systems.items():
            recalls = [per_query_recall(ranked, gold, queries, k, removed) for k in KS]
            cells = " | ".join(f"{sum(r) / len(r):.3f}" for r in recalls)
            if name == setting.reference:
                delta = "(reference)"
            else:
                d, lo, hi = paired_bootstrap(
                    recalls[KS.index(20)], ref_r20, random.Random(0)
                )
                sig = " *" if lo > 0 or hi < 0 else ""
                delta = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"
            print(f"| {name} | {cells} | {delta} |")


def parse_colbert(specs: Sequence[str]) -> dict[str, str]:
    """Parse NAME=PREFIX specs into a mapping."""
    return dict(spec.split("=", 1) for spec in specs)


def main() -> None:
    """Score every setting and print markdown tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colbert", nargs="*", default=[])
    parser.add_argument("--head-size", type=int, default=20)
    parser.add_argument(
        "--corpus-ids",
        type=Path,
        default=Path("data/s2orc-nlp/splits-connected-1k/corpus_ids.txt"),
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("data/s2orc-nlp/citations/references.jsonl"),
    )
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=Path("output/genret/facet_candidates_1k"),
    )
    args = parser.parse_args()

    head = popularity_head(args.references, read_ids(args.corpus_ids), args.head_size)
    weights = sge.load_facet_weights(args.candidates_dir)
    colbert = parse_colbert(args.colbert)

    ev = Path("output/genret/eval")
    in_corpus_decode = ev / "glen-graphfallback_per_bullet.jsonl"
    in_corpus: dict[str, Ranked] = {
        "GRAFT (graph-RRF)": sge.load_eval_graph_rrf(in_corpus_decode, weights),
        "GRAFT (uniform RRF)": load_ranked(in_corpus_decode),
        "Dense-graph": load_ranked(Path("output/eval_1k/retrieval_graph.jsonl")),
        "MiniLM (title+abstract)": load_ranked(
            Path("output/eval_1k/retrieval_minilm.jsonl")
        ),
        "SPECTER2 (title+abstract)": load_ranked(
            Path("output/eval_1k/retrieval_specter2.jsonl")
        ),
        "BM25 (title+facets)": load_ranked(Path("output/eval_1k/retrieval_bm25.jsonl")),
    }
    ooc_sets: dict[str, dict[str, Ranked]] = {}
    for key, directory, decode in (
        ("ooc", "ooc_eval_1k", "ooc-graphfallback_per_bullet.jsonl"),
        ("kcore", "kcore_eval_1k", "kcore-graphfallback_per_bullet.jsonl"),
    ):
        base = Path("output") / directory
        ooc_sets[key] = {
            "GRAFT (mutual corroboration)": sge.load_eval_graph_rrf_mutual(
                ev / decode, weights
            ),
            "GRAFT (uniform RRF)": load_ranked(ev / decode),
            "Dense-graph (gamma=0)": load_ranked(
                base / f"retrieval_densegraph_{key}.jsonl"
            ),
            "MiniLM (title+abstract)": load_ranked(
                base / f"retrieval_minilm_ta_{key}.jsonl"
            ),
            "SPECTER2 (title+abstract)": load_ranked(
                base / f"retrieval_specter2_{key}.jsonl"
            ),
            "BM25 (title+facets)": load_ranked(base / f"retrieval_bm25_{key}.jsonl"),
        }
    for name, prefix in colbert.items():
        label = f"ColBERT {name}"
        in_corpus[label] = load_ranked(Path(f"{prefix}_incorpus.jsonl"))
        for key, systems in ooc_sets.items():
            systems[label] = load_ranked(Path(f"{prefix}_{key}.jsonl"))

    test = Path("data/s2orc-nlp/splits-connected-1k/test.txt")
    settings = [
        Setting(
            name="In-corpus, strict gold",
            gold=Path("output/genret/codebook_training_pairs_1k.jsonl.zst"),
            queries=test,
            systems=in_corpus,
            reference="GRAFT (graph-RRF)",
        ),
        Setting(
            name="In-corpus, direct-citation gold",
            gold=Path("output/ooc_eval_1k/gold_incorpus_cite_pairs.jsonl"),
            queries=test,
            systems=in_corpus,
            reference="GRAFT (graph-RRF)",
        ),
        Setting(
            name="OOC unmatched, citation gold",
            gold=Path("output/ooc_eval_1k/gold_ooc_cite_pairs.jsonl"),
            queries=Path("output/ooc_eval_1k/queries_ooc.txt"),
            systems=ooc_sets["ooc"],
            reference="GRAFT (mutual corroboration)",
        ),
        Setting(
            name="OOC k-core matched, citation gold",
            gold=Path("output/kcore_eval_1k/gold_kcore_cite_pairs.jsonl"),
            queries=Path("output/kcore_eval_1k/queries_kcore.txt"),
            systems=ooc_sets["kcore"],
            reference="GRAFT (mutual corroboration)",
        ),
    ]
    print(f"Popularity head: top {args.head_size} corpus papers by in-degree")
    for setting in settings:
        score_setting(setting, head)


if __name__ == "__main__":
    main()
