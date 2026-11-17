"""Embedding similarity baselines for paper retrieval.

Embeds title+abstract with a sentence-transformer model and retrieves
candidates by cosine similarity. Serves as a dense baseline without
facet extraction, graph construction, or reranking.
"""

# pyright: basic
import importlib
import json
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from genret._shed.transformers import load_tokenizer
from genret.paper_parsing import PaperGraph
from genret.processing.facet_graph_eval import compute_metrics, load_ground_truth
from genret.types import CorpusId
from genret.utils import iter_jsonl, open_auto, track_progress

logger = logging.getLogger(__name__)


def load_paper_texts(
    paper_graphs_path: Path,
) -> tuple[list[CorpusId], list[str]]:
    """Load title+abstract text for each paper.

    Args:
        paper_graphs_path: Path to paper_graphs JSONL file.

    Returns:
        Tuple of (corpus_ids, texts) in the same order.
    """
    corpus_ids: list[CorpusId] = []
    texts: list[str] = []

    for record in track_progress(iter_jsonl(paper_graphs_path), "Loading papers..."):
        paper = PaperGraph.model_validate(record)
        cid = paper.corpusid
        corpus_ids.append(cid)
        texts.append(f"{paper.title} {paper.abstract or ''}".strip())

    logger.info(
        f"Loaded {len(corpus_ids)} papers, "
        f"{sum(1 for t in texts if ' ' in t and len(t) > 100)} with abstract"
    )
    return corpus_ids, texts


SPECTER2_MODELS = {"allenai/specter2", "specter2"}
"""Model names that trigger SPECTER2 adapter loading instead of sentence-transformers."""


def _embed_specter2(
    texts: list[str],
    batch_size: int = 64,
    device: str | None = None,
) -> npt.NDArray[np.float32]:
    """Embed texts using SPECTER2 with the proximity adapter.

    Uses CLS pooling and the adapters library as required by SPECTER2.

    Args:
        texts: List of text strings to embed.
        batch_size: Texts per forward pass.
        device: Device to use (None = auto-detect).

    Returns:
        L2-normalised embedding matrix of shape (len(texts), D).
    """
    import itertools  # noqa: PLC0415

    import torch  # noqa: PLC0415

    # 'adapters' was dropped from the training extra (it pins
    # transformers<4.58). SPECTER-2 baselines need it installed ad hoc.
    auto_adapter_model = importlib.import_module("adapters").AutoAdapterModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer("allenai/specter2_base")
    model = auto_adapter_model.from_pretrained("allenai/specter2_base")
    model.load_adapter(
        "allenai/specter2", source="hf", load_as="specter2", set_active=True
    )
    model = model.to(device)
    model.eval()

    dim = model.config.hidden_size
    logger.info(f"Loaded SPECTER2 + proximity adapter (dim={dim}) on {device}")

    all_embeddings: list[npt.NDArray[np.float32]] = []
    for batch in track_progress(
        list(itertools.batched(texts, batch_size, strict=False)),
        "Embedding (SPECTER2)...",
    ):
        encoded = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        # CLS pooling
        cls_embeddings = output.last_hidden_state[:, 0, :]
        # L2 normalise
        norms = cls_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        normalised = cls_embeddings / norms
        all_embeddings.append(normalised.cpu().numpy().astype(np.float32))

    return np.concatenate(all_embeddings, axis=0)


def _embed_sentence_transformer(
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    device: str | None = None,
) -> npt.NDArray[np.float32]:
    """Embed texts using SPECTER2 with the proximity adapter.

    Uses CLS pooling and the adapters library as required by SPECTER2.

    Args:
        texts: List of text strings to embed.
        model_name: HuggingFace model identifier.
        batch_size: Texts per forward pass.
        device: Device to use (None = auto-detect).

    Returns:
        L2-normalised embedding matrix of shape (len(texts), D).
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(model_name, device=device)
    logger.info(f"Loaded {model_name} (dim={model.get_sentence_embedding_dimension()})")

    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    device: str | None = None,
) -> npt.NDArray[np.float32]:
    """Embed texts using a sentence-transformers model or SPECTER2.

    Args:
        texts: List of text strings to embed.
        model_name: HuggingFace model ID.
        batch_size: Texts per forward pass.
        device: Device to use (None = auto-detect).

    Returns:
        L2-normalised embedding matrix of shape (len(texts), D).
    """
    if model_name in SPECTER2_MODELS:
        return _embed_specter2(texts, batch_size=batch_size, device=device)
    else:
        return _embed_sentence_transformer(
            texts, model_name, batch_size=batch_size, device=device
        )


def retrieve_by_similarity(
    embeddings: npt.NDArray[np.float32],
    corpus_ids: list[CorpusId],
    query_ids: set[CorpusId],
    top_n: int = 50,
    device: str | None = None,
) -> dict[CorpusId, list[dict[str, float | int]]]:
    """Retrieve top-N papers by cosine similarity.

    Computes only query-vs-corpus similarities (not the full NxN matrix)
    and uses GPU when available.

    Args:
        embeddings: L2-normalised embedding matrix (N, D).
        corpus_ids: Corpus IDs matching embedding rows.
        query_ids: Set of query corpus IDs.
        top_n: Number of results per query.
        device: Device for similarity computation (None = auto-detect).

    Returns:
        Dict mapping query corpus ID to ranked result list.
    """
    import torch  # noqa: PLC0415

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    # Build query index mapping
    sorted_qids = sorted(query_ids)
    query_indices = [cid_to_idx[qid] for qid in sorted_qids if qid in cid_to_idx]
    valid_qids = [qid for qid in sorted_qids if qid in cid_to_idx]

    # Compute query-vs-corpus similarities on GPU
    logger.info("Computing query-corpus similarities on %s...", device)
    all_emb = torch.from_numpy(embeddings).to(device)
    query_emb = all_emb[query_indices]  # (Q, D)
    sim = query_emb @ all_emb.T  # (Q, N)

    # Exclude self-retrieval
    for i, idx in enumerate(query_indices):
        sim[i, idx] = -1.0

    # Top-k per query
    top_scores, top_idx = torch.topk(sim, top_n, dim=1)
    top_scores_np = top_scores.cpu().numpy()
    top_idx_np = top_idx.cpu().numpy()

    return {
        qid: [
            {
                "corpusid": int(corpus_ids[top_idx_np[i, j]]),
                "score": float(top_scores_np[i, j]),
            }
            for j in range(top_n)
        ]
        for i, qid in enumerate(valid_qids)
    }


def _evaluate_and_log(
    result_lists: dict[int, list[int]], co_cit_path: Path, bib_path: Path
) -> None:
    """Evaluate retrieval results and log metrics."""
    valid_ids = frozenset(result_lists)
    combined_gt, co_cit_gt, bib_gt = load_ground_truth(co_cit_path, bib_path, valid_ids)

    for label, gt in [
        ("Combined", combined_gt),
        ("Co-citation", co_cit_gt),
        ("Bib-coupling", bib_gt),
    ]:
        metrics = compute_metrics(result_lists, gt)
        logger.info(
            f"{label}: Recall@5={metrics.recall[5]:.4f}, "
            f"Recall@10={metrics.recall[10]:.4f}, "
            f"Recall@20={metrics.recall[20]:.4f}, "
            f"MRR={metrics.mrr:.4f} "
            f"({metrics.num_queries} queries, {metrics.num_gt_pairs} GT pairs)"
        )


def run_embedding_baseline(
    paper_graphs_path: Path,
    query_ids_path: Path,
    output_path: Path,
    co_cit_path: Path,
    bib_path: Path,
    model_name: str,
    *,
    top_n: int = 50,
    batch_size: int = 64,
    device: str | None = None,
) -> None:
    """Run embedding similarity baseline and evaluate.

    Args:
        paper_graphs_path: Path to paper_graphs JSONL (with abstracts).
        query_ids_path: Path to file with query corpus IDs (one per line).
        output_path: Path to write retrieval results JSONL.
        co_cit_path: Path to co_citations JSONL.
        bib_path: Path to bib_coupling JSONL.
        model_name: HuggingFace sentence-transformer model ID.
        top_n: Number of candidates per query.
        batch_size: Embedding batch size.
        device: Device for inference.
    """
    corpus_ids, texts = load_paper_texts(paper_graphs_path)

    embeddings = embed_texts(texts, model_name, batch_size=batch_size, device=device)

    # Load query IDs from split file (one corpus ID per line)
    query_ids: set[CorpusId] = set()
    with open(query_ids_path) as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                query_ids.add(CorpusId(int(stripped)))
    logger.info(f"Loaded {len(query_ids)} query IDs")

    retrieval_sets = retrieve_by_similarity(
        embeddings, corpus_ids, query_ids, top_n=top_n, device=device
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_auto(output_path, "w") as fh:
        for qid, results in retrieval_sets.items():
            record = {"corpusid": int(qid), "results": results}
            fh.write(json.dumps(record) + "\n")
    logger.info(f"Wrote {len(retrieval_sets)} retrieval sets to {output_path}")

    result_lists = {
        int(qid): [int(r["corpusid"]) for r in results]
        for qid, results in retrieval_sets.items()
    }
    _evaluate_and_log(result_lists, co_cit_path, bib_path)
