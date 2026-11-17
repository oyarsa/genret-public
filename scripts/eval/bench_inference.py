"""Inference cost of the dense-graph teacher vs GRAFT, end to end from facet text.

For each system: load once, then for every query paper run the full retrieval
path a deployment would run given the query's extracted facet items. Report
per-query latency (median, mean), throughput, peak GPU memory and the size of
what each system needs on disk beyond the model weights.

- Teacher: MiniLM encodes the query's facet items, then `retrieve_new_paper`
  (mean-pool dense search, citation signal, 2-hop BFS, RRF) with the production
  recipe (alpha=0.75, gamma=0.25).
- GRAFT: per-bullet trie-constrained beam search (beam 50) on the fine-tuned 1B
  model, then graph-RRF over the pruned adjacency.

Both share facet extraction, which is not timed. Run on the cluster via the
`bench_inference` fleche job.

Usage:
    uv run --extra cuda --extra training python scripts/eval/bench_inference.py \
        --output output/colbert_1k/bench_inference.json
"""

import argparse
import json
import logging
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel

from genret._shed.transformers import load_tokenizer
from genret.paper_parsing import FacetRecordType, Facets
from genret.processing.facet_retrieval import load_edge_graphs
from genret.processing.local_embeddings import embed_texts
from genret.processing.online_retrieval import load_corpus_index, retrieve_new_paper
from genret.training._cli_helpers import _load_checkpoint
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_dataset import _build_messages, _facet_text
from genret.training.codebook_evaluate import _fuse_facet_results
from genret.training.glen_decoding import decode_per_facet_glen
from genret.training.glen_docids import load_glen_index, split_bullets
from genret.training.glen_evaluate import load_eval_inputs
from genret.training.glen_snap import MINILM_MODEL_NAME
from genret.types import CorpusId
from genret.utils import load_ids_from_file

logger = logging.getLogger(__name__)

RRF_K = 60
"""RRF constant used by graph-RRF."""


def timed_queries(
    run_query: Callable[[CorpusId], Any], queries: Sequence[CorpusId], warmup: int
) -> dict[str, Any]:
    """Run every query once after `warmup` untimed ones; return latency stats."""
    for qid in queries[:warmup]:
        run_query(qid)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    for qid in queries:
        start = time.perf_counter()
        run_query(qid)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    total = sum(latencies)
    return {
        "queries": len(latencies),
        "latency_median_s": statistics.median(latencies),
        "latency_mean_s": total / len(latencies),
        "latency_p90_s": sorted(latencies)[int(0.9 * len(latencies))],
        "throughput_q_per_s": len(latencies) / total,
        "peak_gpu_mem_gb": (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else 0.0
        ),
    }


def dir_size(path: Path, pattern: str = "*") -> int:
    """Total size in bytes of files matching `pattern` directly under `path`."""
    return sum(p.stat().st_size for p in path.glob(pattern) if p.is_file())


def bench_teacher(
    args: argparse.Namespace,
    facets_map: dict[CorpusId, Facets],
    queries: list[CorpusId],
) -> dict[str, Any]:
    """Time the dense-graph teacher, encoder included."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    tokenizer = load_tokenizer(MINILM_MODEL_NAME)
    encoder = AutoModel.from_pretrained(MINILM_MODEL_NAME).to(device).eval()
    corpus_index = load_corpus_index(
        args.graph_dir,
        args.facets,
        references_path=args.references,
        co_citations_path=args.co_citations,
        bib_coupling_path=args.bib_coupling,
    )
    graphs, _ = load_edge_graphs(args.edge_dir)
    load_s = time.perf_counter() - t0

    def run_query(qid: CorpusId) -> None:
        facets = facets_map[qid]
        embeddings = {
            name: embed_texts(
                list(getattr(facets, name)), encoder, tokenizer, device=device
            )
            for name in FACET_NAMES
            if getattr(facets, name)
        }
        retrieve_new_paper(
            query_facets=facets,
            query_embeddings=embeddings,
            corpus_index=corpus_index,
            edge_graphs=graphs,
            query_references=corpus_index.corpus_references.get(qid),
            alpha=0.75,
            beta=0.0,
            gamma=0.25,
            top_n=50,
        )

    stats = timed_queries(run_query, queries, args.warmup)
    stats["load_s"] = load_s
    stats["encoder_params_m"] = sum(p.numel() for p in encoder.parameters()) / 1e6
    stats["index_bytes"] = {
        "facet_item_vectors": dir_size(args.graph_dir, "facet_graph_*.npz"),
        "edges": dir_size(args.edge_dir, "paper_edges_*.npz"),
    }
    encoder.cpu()
    torch.cuda.empty_cache()
    return stats


def bench_graft(
    args: argparse.Namespace,
    facets_map: dict[CorpusId, Facets],
    queries: list[CorpusId],
) -> dict[str, Any]:
    """Time GRAFT: per-bullet constrained decoding plus graph-RRF."""
    t0 = time.perf_counter()
    model, tokeniser = _load_checkpoint(args.checkpoint)
    index = load_glen_index(args.glen_dir)
    weights: dict[str, dict[int, dict[int, float]]] = {}
    for facet in FACET_NAMES:
        f = np.load(args.candidates_dir / f"facet_candidates_{facet}.npz")
        cids = f["corpusids"].astype("int64")
        cands = f["candidates"].astype("int64")
        scores = f["scores_combined"].astype("float32")
        weights[facet] = {
            int(cids[i]): {
                int(cids[j]): float(scores[i, pos])
                for pos, j in enumerate(cands[i])
                if j >= 0
            }
            for i in range(len(cids))
        }
    load_s = time.perf_counter() - t0

    def run_query(qid: CorpusId) -> None:
        facets = facets_map[qid]
        per_facet: dict[str, list[Any]] = {}
        for facet in FACET_NAMES:
            text = _facet_text(facets, facet, None)
            bullets = split_bullets(text) or ([text] if text else [])
            beams: list[Any] = []
            for bullet in bullets:
                beams.extend(
                    decode_per_facet_glen(
                        model=model,
                        tokeniser=tokeniser,
                        messages=_build_messages(facet, bullet, kind="retrieval"),
                        facet=facet,
                        index=index,
                        num_beams=args.beams,
                        max_new_tokens=args.max_docid_tokens,
                    )
                )
            per_facet[facet] = beams
        fused = _fuse_facet_results(per_facet)
        # graph-RRF: rescale each facet's reciprocal-rank term by the edge weight
        scored: dict[int, float] = {}
        for r in fused:
            s = 0.0
            for facet, rank in r.facet_ranks.items():
                w = weights.get(facet, {}).get(int(qid), {}).get(int(r.corpusid), 0.0)
                if w > 0:
                    s += w / (RRF_K + rank)
            scored[int(r.corpusid)] = s
        sorted(scored.items(), key=lambda x: -x[1])

    stats = timed_queries(run_query, queries, args.warmup)
    stats["load_s"] = load_s
    stats["model_params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
    stats["index_bytes"] = {
        "trie_and_code_to_papers": dir_size(args.glen_dir, "*.json"),
        "adjacency": dir_size(args.candidates_dir, "facet_candidates_*.npz"),
    }
    return stats


def main() -> None:
    """Benchmark both systems on the in-corpus test queries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/s2orc-nlp/splits-connected-1k/test.txt"),
    )
    parser.add_argument(
        "--facets",
        type=Path,
        default=Path("data/s2orc-nlp/splits-connected-1k/processed/facets.jsonl.zst"),
    )
    parser.add_argument(
        "--graph-dir", type=Path, default=Path("output/eval_1k/graph_prod")
    )
    parser.add_argument(
        "--edge-dir", type=Path, default=Path("output/eval_1k/edges_prod_dc")
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("data/s2orc-nlp/citations/references.jsonl"),
    )
    parser.add_argument(
        "--co-citations",
        type=Path,
        default=Path("data/s2orc-nlp/citations/co_citations.jsonl"),
    )
    parser.add_argument(
        "--bib-coupling",
        type=Path,
        default=Path("data/s2orc-nlp/citations/bib_coupling.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "output/genret/checkpoints/glen-bullets-1k-1b-optionB-graphfallback"
        ),
    )
    parser.add_argument(
        "--glen-dir", type=Path, default=Path("output/genret/glen_index_1k")
    )
    parser.add_argument(
        "--candidates-dir", type=Path, default=Path("output/genret/facet_candidates_1k")
    )
    parser.add_argument("--beams", type=int, default=50)
    parser.add_argument("--max-docid-tokens", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--systems", nargs="+", default=["teacher", "graft"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    facets_map, _ = load_eval_inputs(args.facets, FacetRecordType.STANDARD)
    queries = sorted(load_ids_from_file(args.queries))
    if args.limit:
        queries = queries[: args.limit]

    results: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "queries": len(queries),
    }
    if "teacher" in args.systems:
        results["teacher"] = bench_teacher(args, facets_map, queries)
        logger.info("teacher: %s", results["teacher"])
    if "graft" in args.systems:
        results["graft"] = bench_graft(args, facets_map, queries)
        logger.info("graft: %s", results["graft"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
