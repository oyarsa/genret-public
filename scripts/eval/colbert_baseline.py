"""ColBERT late-interaction baseline for related-paper retrieval.

Encodes title+abstract of every corpus paper as a ColBERT document and every
query paper as a ColBERT query, then ranks the whole corpus by exact MaxSim (no
PLAID index; the 1k corpus is small enough). Works for zero-shot checkpoints
and for checkpoints fine-tuned by `scripts/colbert/train_colbert.py`.

Output files use the standard retrieval JSONL format, so they are scored with
`scripts/eval/score_glen_eval.py`. PyLate is not a project dependency, so run:

    uv run --no-project --python 3.14 --with-editable . --with pylate
      python scripts/eval/colbert_baseline.py ...
"""

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from genret._shed.pylate import encode_multivector, load_colbert
from genret.utils import iter_jsonl

logger = logging.getLogger(__name__)


def load_texts(paper_graphs_path: Path) -> dict[int, str]:
    """Load title+abstract text per paper, formatted as the MiniLM baseline.

    Args:
        paper_graphs_path: paper_graphs JSONL with `corpusid`, `title` and
            `abstract` fields.

    Returns:
        Mapping from corpus ID to "title abstract" text.
    """
    texts: dict[int, str] = {}
    for record in iter_jsonl(paper_graphs_path):
        title: str = record.get("title") or ""
        abstract: str = record.get("abstract") or ""
        texts[int(record["corpusid"])] = f"{title} {abstract}".strip()
    return texts


def read_ids(path: Path) -> list[int]:
    """Read one corpus ID per line, skipping blank lines."""
    return [int(line) for line in path.read_text().split()]


def pad_documents(
    embeddings: Sequence[npt.NDArray[np.float32]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length document embeddings into a padded tensor.

    Args:
        embeddings: One `(num_tokens, dim)` array per document.

    Returns:
        Tuple of `(docs, mask)` with shapes `(D, L, dim)` and `(D, L)`, where
        mask is True for real tokens.
    """
    max_len = max(e.shape[0] for e in embeddings)
    dim = embeddings[0].shape[1]
    docs = torch.zeros((len(embeddings), max_len, dim), dtype=torch.float32)
    mask = torch.zeros((len(embeddings), max_len), dtype=torch.bool)
    for i, emb in enumerate(embeddings):
        docs[i, : emb.shape[0]] = torch.from_numpy(emb)
        mask[i, : emb.shape[0]] = True
    return docs, mask


def maxsim_scores(
    query: npt.NDArray[np.float32], docs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Score one query against every document with ColBERT MaxSim.

    Args:
        query: Query token embeddings `(Lq, dim)`.
        docs: Padded document embeddings `(D, Ld, dim)`.
        mask: Document token mask `(D, Ld)`.

    Returns:
        Scores of shape `(D,)`: sum over query tokens of the max similarity to
        any real document token.
    """
    sim = torch.einsum("qh,dlh->dql", torch.from_numpy(query), docs)
    sim = sim.masked_fill(~mask[:, None, :], float("-inf"))
    return sim.max(dim=2).values.sum(dim=1)


def retrieve(
    query_embeddings: Mapping[int, npt.NDArray[np.float32]],
    corpus_ids: Sequence[int],
    docs: torch.Tensor,
    mask: torch.Tensor,
    top_n: int,
) -> list[dict[str, int | list[dict[str, int | float]]]]:
    """Rank the corpus for each query, excluding the query paper itself.

    Args:
        query_embeddings: Query token embeddings by query corpus ID.
        corpus_ids: Corpus IDs aligned with the rows of `docs`.
        docs: Padded document embeddings.
        mask: Document token mask.
        top_n: Results per query.

    Returns:
        Retrieval records in the standard JSONL format.
    """
    records: list[dict[str, int | list[dict[str, int | float]]]] = []
    for qid, query in query_embeddings.items():
        scores = maxsim_scores(query, docs, mask)
        order = torch.argsort(scores, descending=True).tolist()
        results: list[dict[str, int | float]] = []
        for idx in order:
            if corpus_ids[idx] == qid:
                continue
            results.append({"corpusid": corpus_ids[idx], "score": float(scores[idx])})
            if len(results) == top_n:
                break
        records.append({"corpusid": qid, "results": results})
    return records


def main() -> None:
    """Encode the corpus once and write one retrieval file per query set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-graphs", type=Path, required=True)
    parser.add_argument("--corpus-ids", type=Path, required=True)
    parser.add_argument(
        "--queries",
        nargs="+",
        required=True,
        help="Query sets as ID_FILE=OUTPUT_JSONL pairs.",
    )
    parser.add_argument("--model", default="lightonai/colbertv2.0")
    parser.add_argument("--query-length", type=int, default=300)
    parser.add_argument("--document-length", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    texts = load_texts(args.paper_graphs)
    corpus_ids = read_ids(args.corpus_ids)
    model = load_colbert(
        args.model,
        query_length=args.query_length,
        document_length=args.document_length,
        device=args.device,
    )

    logger.info("Encoding %d corpus documents", len(corpus_ids))
    doc_embeddings = encode_multivector(
        model,
        [texts[cid] for cid in corpus_ids],
        is_query=False,
        batch_size=args.batch_size,
    )
    docs, mask = pad_documents(doc_embeddings)

    for spec in args.queries:
        ids_path, output_path = (Path(p) for p in spec.split("=", 1))
        query_ids = read_ids(ids_path)
        logger.info("Encoding %d queries from %s", len(query_ids), ids_path)
        query_embeddings = encode_multivector(
            model,
            [texts[qid] for qid in query_ids],
            is_query=True,
            batch_size=args.batch_size,
        )
        records = retrieve(
            dict(zip(query_ids, query_embeddings, strict=True)),
            corpus_ids,
            docs,
            mask,
            args.top_n,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        logger.info("Wrote %d retrieval sets to %s", len(records), output_path)


if __name__ == "__main__":
    main()
