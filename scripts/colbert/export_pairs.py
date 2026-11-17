"""Export GRAFT's training supervision as paper-level pairs for ColBERT.

Rebuilds the exact per-bullet dataset GRAFT (5b3h) was trained on with
`GlenBulletDataset`, then maps every example back to its (query paper, target
paper) pair: the query paper from the title in the prompt, the target paper
from the target bullet text. Writes one JSONL row per pair with `count`, the
number of GRAFT training examples behind it (so a replicated variant can
reproduce GRAFT's exposure distribution and a deduplicated variant can ignore
it).

Examples whose title or target bullet maps to more than one paper are dropped
and counted.
"""

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from genret.paper_parsing import FacetRecordType, facets_from_type
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_dataset import _facet_text
from genret.training.glen_bullet_dataset import (
    GlenBulletDataset,
    GlenBulletDatasetConfig,
)
from genret.training.glen_docids import normalise_facet_text, split_bullets
from genret.types import CorpusId
from genret.utils import iter_jsonl, load_ids_from_file

logger = logging.getLogger(__name__)

GRAFT_EXAMPLE_COUNT = 294_089
"""Example count logged for the 5b3h training run (EXPERIMENT_LOG)."""
GRAFT_SEED = 42
"""Dataset RNG seed used by `train-bullets` (TrainingConfig default)."""


def build_title_index(paper_graphs_path: Path) -> dict[str, set[CorpusId]]:
    """Map stripped titles to the papers that carry them."""
    index: dict[str, set[CorpusId]] = defaultdict(set)
    for record in iter_jsonl(paper_graphs_path):
        title = record.get("title")
        if isinstance(title, str) and title.strip():
            index[title.strip()].add(CorpusId(int(record["corpusid"])))
    return index


def build_bullet_index(facets_path: Path) -> dict[tuple[str, str], set[CorpusId]]:
    """Map (facet, normalised bullet text) to the papers that carry it."""
    index: dict[tuple[str, str], set[CorpusId]] = defaultdict(set)
    for raw in iter_jsonl(facets_path):
        facets = facets_from_type(raw, FacetRecordType.STANDARD)
        for facet in FACET_NAMES:
            for bullet in split_bullets(_facet_text(facets, facet, None)):
                key = (facet, normalise_facet_text(bullet))
                index[key].add(facets.corpusid)
    return index


def parse_prompt(user_content: str) -> tuple[str, str]:
    """Extract (title, facet) from a GRAFT bullet prompt.

    Args:
        user_content: User message: a title line, a facet line, then bullets.

    Returns:
        The title and the lower-cased facet name.
    """
    title_line, facet_line, *_ = user_content.split("\n")
    return title_line.removeprefix("Title: ").strip(), facet_line.rstrip(":").lower()


def main() -> None:
    """Rebuild GRAFT's dataset and write paper-level pair counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("output/genret/codebook_training_pairs_1k.jsonl.zst"),
    )
    parser.add_argument(
        "--facets",
        type=Path,
        default=Path("data/s2orc-nlp/splits-connected-1k/processed/facets.jsonl.zst"),
    )
    parser.add_argument(
        "--paper-graphs",
        type=Path,
        default=Path(
            "data/s2orc-nlp/splits-connected-1k/processed/paper_graphs.jsonl.zst"
        ),
    )
    parser.add_argument(
        "--train-split",
        type=Path,
        default=Path("data/s2orc-nlp/splits-connected-1k/train.txt"),
    )
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=Path("output/genret/facet_candidates_1k_goldsource"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dataset = GlenBulletDataset(
        GlenBulletDatasetConfig(
            pairs_path=args.pairs,
            facets_path=args.facets,
            paper_graphs_path=args.paper_graphs,
            split_ids=load_ids_from_file(args.train_split),
            graph_sampling="weighted",
            candidates_dir=args.candidates_dir,
            min_target_views=3,
            min_views_graph_fallback=10,
        ),
        rng=random.Random(GRAFT_SEED),
    )
    n_examples = len(dataset)
    logger.info(
        "Rebuilt %d examples (5b3h logged %d, diff %+d)",
        n_examples,
        GRAFT_EXAMPLE_COUNT,
        n_examples - GRAFT_EXAMPLE_COUNT,
    )

    titles = build_title_index(args.paper_graphs)
    bullets = build_bullet_index(args.facets)

    counts: Counter[tuple[CorpusId, CorpusId]] = Counter()
    failures: Counter[str] = Counter()
    for i in range(n_examples):
        example = dataset[i]
        title, facet = parse_prompt(example.prompt_messages[1]["content"])
        queries = titles.get(title, set())
        targets = bullets.get(
            (facet, normalise_facet_text(example.target_docid)), set()
        )
        if len(queries) != 1:
            failures[f"query_{'missing' if not queries else 'ambiguous'}"] += 1
            continue
        if len(targets) != 1:
            failures[f"target_{'missing' if not targets else 'ambiguous'}"] += 1
            continue
        (query,) = queries
        (target,) = targets
        counts[query, target] += 1

    logger.info(
        "Mapped %d examples to %d unique pairs; dropped %s",
        sum(counts.values()),
        len(counts),
        dict(failures),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for (query, target), count in sorted(counts.items()):
            row = {"query": int(query), "positive": int(target), "count": count}
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
