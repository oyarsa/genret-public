# pyright: basic
"""Re-rankers for facet-based candidate pairs.

Provides a pair formatter that turns two papers (title + facets) into a single
text string, and two re-ranker implementations:

- ``LocalReranker``: local HuggingFace cross-encoder (ModernBERT)
- ``LLMReranker``: pairwise scoring via an OpenAI LLM (GPT-5.4 Mini/Nano)

Both inherit from the ``Reranker`` ABC.
"""

import asyncio
import itertools
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase
    from transformers.modeling_outputs import SequenceClassifierOutput

from genret.paper_parsing import Facets
from genret.paper_parsing import load_facets_map as load_facets_map
from genret.processing.facet_evaluation import FACET_NAMES
from genret.types import CorpusId
from genret.utils import iter_jsonl, resolve_jsonl, track_progress

logger = logging.getLogger(__name__)

FACET_LABELS = {
    "problems": "Problems",
    "methods": "Methods",
    "results": "Results",
    "contributions": "Contributions",
}
"""Display labels for each facet category."""


def facet_items(facets: Facets, name: str) -> Sequence[str]:
    """Return the item list for a given facet name.

    Args:
        facets: Paper facets record.
        name: One of the canonical facet names.

    Returns:
        Sequence of facet item strings.
    """
    if name == "problems":
        return facets.problems
    if name == "methods":
        return facets.methods
    if name == "results":
        return facets.results
    if name == "contributions":
        return facets.contributions

    raise ValueError(f"Unknown facet name: {name}")


def _format_paper(title: str, facets: Facets) -> str:
    """Format a single paper as a titled block with non-empty facet lines.

    Args:
        title: Paper title.
        facets: Paper facets record.

    Returns:
        Multi-line string with the paper title and its facet items.
    """
    lines = [title]

    for name in FACET_NAMES:
        if items := facet_items(facets, name):
            lines.append(f"{FACET_LABELS[name]}: {'; '.join(items)}")

    return "\n".join(lines)


def format_paper_pair(
    title_a: str, facets_a: Facets, title_b: str, facets_b: Facets
) -> str:
    """Format two papers as a cross-encoder input string.

    Each paper is rendered as a titled block with its non-empty facet lines.
    The two blocks are separated by a blank line.

    Args:
        title_a: Title of the first paper.
        facets_a: Facets of the first paper.

        title_b: Title of the second paper.
        facets_b: Facets of the second paper.

    Returns:
        Formatted pair string ready for tokenisation.
    """
    block_a = _format_paper(title_a, facets_a)
    block_b = _format_paper(title_b, facets_b)
    return f"Paper A: {block_a}\n\nPaper B: {block_b}"


class Reranker(ABC):
    """Base class for all re-ranker implementations."""

    @abstractmethod
    def score_pairs(
        self, texts: list[str], batch_size: int = 32
    ) -> npt.NDArray[np.float32]:
        """Score formatted paper pair strings.

        Args:
            texts: List of formatted pair strings from ``format_paper_pair``.
            batch_size: Implementation-specific batch parameter.

        Returns:
            1-D array of scores (one per input text); higher means more related.
        """

    @abstractmethod
    def inference_batch_size(self) -> int:
        """Return the recommended batch size for this re-ranker."""


@dataclass(frozen=True, kw_only=True)
class RerankerConfig:
    """Configuration for loading and running a cross-encoder re-ranker."""

    model_name: str
    """HuggingFace model ID or local checkpoint path."""
    max_length: int = 2048
    """Maximum token length for the tokeniser."""
    device: str = "cuda"
    """Device to run inference on (`"cuda"` or `"cpu"`)."""


class LocalReranker(Reranker):
    """Thin wrapper around a HuggingFace sequence-classification model.

    Provides batched scoring of pre-formatted paper pair strings, returning
    raw logits (higher = more related).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: RerankerConfig,
    ) -> None:
        """Initialise with a loaded model, tokeniser, and config.

        Args:
            model: HuggingFace sequence-classification model.
            tokenizer: Matching tokeniser.
            config: Re-ranker configuration.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._config = config

    @classmethod
    def from_pretrained(cls, config: RerankerConfig) -> LocalReranker:
        """Load a pre-trained or fine-tuned cross-encoder.

        Args:
            config: Re-ranker configuration specifying the model path and device.

        Returns:
            Ready-to-use re-ranker model.
        """
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
            PreTrainedModel,
            PreTrainedTokenizerBase,
        )

        # Try loading tokeniser from the model path. If not found (e.g.
        # intermediate checkpoint), try the parent directory where
        # `train_distilled` saves the tokeniser.
        tokenizer_path = config.model_name
        model_path = Path(config.model_name)
        if model_path.is_dir() and not (model_path / "tokenizer_config.json").exists():
            parent = model_path.parent
            if (parent / "tokenizer_config.json").exists():
                tokenizer_path = str(parent)
                logger.info("Tokeniser not in %s, using parent %s", model_path, parent)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError(f"Unexpected tokenizer type: {type(tokenizer)}")

        # Try loading with num_labels=1 first (reranker/regression models).
        # If that fails (e.g. NLI models with 3 labels), load with the
        # model's native label count.
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name, num_labels=1
            )
        except RuntimeError:
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name
            )
        if not isinstance(model, PreTrainedModel):
            raise TypeError(f"Unexpected model type: {type(model)}")

        import torch  # noqa: PLC0415

        from genret._shed.torch import model_to_device  # noqa: PLC0415

        torch.set_float32_matmul_precision("high")
        model = model_to_device(model, config.device)
        model.eval()

        logger.info(f"Loaded re-ranker from {config.model_name} on {config.device}")
        return cls(model=model, tokenizer=tokenizer, config=config)

    @override
    def score_pairs(
        self, texts: list[str], batch_size: int = 32
    ) -> npt.NDArray[np.float32]:
        """Score formatted paper pair strings.

        Args:
            texts: List of formatted pair strings from `format_paper_pair`.
            batch_size: Number of pairs to process per forward pass.

        Returns:
            1-D array of logits (one per input text); higher means more related.
        """
        all_logits: list[float] = []

        for batch_texts in itertools.batched(texts, batch_size, strict=False):
            inputs: BatchEncoding = self._tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=self._config.max_length,
                return_tensors="pt",
            ).to(self._config.device)

            outputs: SequenceClassifierOutput = self._model(**inputs)
            assert outputs.logits is not None
            logits = outputs.logits
            # For regression models (num_labels=1) logits are [batch, 1].
            # For classification (num_labels>1) use the last column as the
            # positive-class score.
            if logits.ndim == 2 and logits.shape[1] > 1:
                from genret._shed.torch import tensor_column  # noqa: PLC0415

                scores = tensor_column(logits, -1)
            else:
                scores = logits.squeeze(-1)
            all_logits.extend(scores.tolist())

        return np.array(all_logits, dtype=np.float32)

    @override
    def inference_batch_size(self) -> int:
        """Return a safe default inference batch size for the current device."""
        if self._config.device == "cpu":
            return 4
        return 32


_LLM_RERANK_SYSTEM = """\
You are a relevance judge for academic papers. You will be given two papers, \
each with a title and facets (problems, methods, results, contributions). \
Rate how related the papers are on a scale from 0 to 10, where 0 means \
completely unrelated and 10 means nearly identical in topic and approach."""


class _RelevanceScore(BaseModel, frozen=True):
    """Structured output for LLM-based pairwise relevance scoring."""

    reasoning: str
    """Brief explanation of the relevance judgement."""
    score: float
    """Relevance score from 0 (unrelated) to 10 (nearly identical)."""


class LLMReranker(Reranker):
    """Pairwise re-ranker backed by an OpenAI LLM.

    Sends each paper pair as a structured-output request and asks the model
    to rate relevance on a 0-10 scale. Pairs are scored concurrently using
    the existing ``OpenAIClient.batch_complete`` infrastructure.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        concurrent: int = 30,
    ) -> None:
        """Initialise the LLM re-ranker.

        Args:
            model: Model name (e.g. ``"gpt-5.4-nano"``). For OpenRouter,
                use their model IDs (e.g. ``"mistralai/mistral-small-3.2"``).
            base_url: Override the API base URL (e.g. OpenRouter).
            api_key: Override the API key. None falls back to env var.
            concurrent: Maximum number of concurrent API requests.
        """
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._concurrent = concurrent

    @override
    def score_pairs(
        self, texts: list[str], batch_size: int = 32
    ) -> npt.NDArray[np.float32]:
        """Score formatted paper pair strings via LLM.

        Called from a worker thread (via ``asyncio.to_thread``), so it is safe
        to create and run a new event loop here.

        Args:
            texts: List of formatted pair strings from ``format_paper_pair``.
            batch_size: Ignored (concurrency is controlled by ``concurrent``).

        Returns:
            1-D array of relevance scores (0-10), one per input text.
        """
        return asyncio.run(self._score_pairs_async(texts))

    async def _score_pairs_async(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Score all pairs concurrently."""
        from genret.llm.client import CompletionRequest, OpenAIClient  # noqa: PLC0415

        client = OpenAIClient(
            model=self._model,
            base_url=self._base_url,
            api_key=self._api_key,
            temperature=0.0,
            timeout=30.0,
        )

        requests = (
            (i, CompletionRequest(system=_LLM_RERANK_SYSTEM, user=text))
            for i, text in enumerate(texts)
        )

        scores = [0.0] * len(texts)
        total_cost = 0.0
        async for idx, result in client.batch_complete(
            requests,
            response_model=_RelevanceScore,
            concurrent=self._concurrent,
        ):
            total_cost += result.cost
            if result.result is not None:
                scores[idx] = max(0.0, min(10.0, result.result.score))
            else:
                logger.warning("LLM rerank: failed to score pair %d", idx)

        await client.close()
        logger.info(
            "LLM rerank: scored %d pairs with %s, cost $%.4f",
            len(texts),
            self._model,
            total_cost,
        )
        return np.array(scores, dtype=np.float32)

    @override
    def inference_batch_size(self) -> int:
        """Return batch size (unused, concurrency is internal)."""
        return self._concurrent


def load_titles(paper_graphs_path: Path) -> dict[CorpusId, str]:
    """Load paper titles from paper_graphs JSONL.

    Only papers with non-empty titles are included.

    Args:
        paper_graphs_path: Path to paper_graphs JSONL file.

    Returns:
        Mapping from corpus ID to title string.
    """
    resolved = resolve_jsonl(paper_graphs_path)
    titles: dict[CorpusId, str] = {}
    for record in track_progress(iter_jsonl(resolved), desc="Loading titles..."):
        cid = CorpusId(int(record["corpusid"]))
        title = record.get("title", "")
        if isinstance(title, str) and title.strip():
            titles[cid] = title.strip()

    logger.info(f"Loaded {len(titles)} titles")
    return titles


def _format_candidate_pairs(
    corpusids: npt.NDArray[np.int64],
    candidates: npt.NDArray[np.int32],
    facets_map: dict[CorpusId, Facets],
    titles: dict[CorpusId, str],
) -> tuple[list[str], list[tuple[int, int]]]:
    """Format all valid candidate pairs as cross-encoder input strings.

    Args:
        corpusids: 1-D array of corpus IDs.
        candidates: (N, K) array of candidate indices into `corpusids`.
        facets_map: Corpus ID to facets mapping.
        titles: Corpus ID to title mapping.

    Returns:
        Tuple of (formatted text strings, corresponding (row, col) indices).
    """
    n = len(corpusids)
    all_texts: list[str] = []
    all_indices: list[tuple[int, int]] = []

    for i in range(n):
        cid_i = CorpusId(int(corpusids[i]))
        if cid_i not in facets_map or cid_i not in titles:
            continue

        for j_col in range(candidates.shape[1]):
            j = int(candidates[i, j_col])
            if j < 0 or j >= n:
                continue

            cid_j = CorpusId(int(corpusids[j]))
            if cid_j not in facets_map or cid_j not in titles:
                continue

            text = format_paper_pair(
                titles[cid_i], facets_map[cid_i], titles[cid_j], facets_map[cid_j]
            )
            all_texts.append(text)
            all_indices.append((i, j_col))

    return all_texts, all_indices


def _select_top_k(
    candidates: npt.NDArray[np.int32],
    scores: npt.NDArray[np.float32],
    all_indices: list[tuple[int, int]],
    n: int,
    top_k: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Select the top-K candidates per paper by re-ranker score.

    Args:
        candidates: (N, K_in) array of candidate indices from the input NPZ.
        scores: 1-D array of re-ranker scores, one per entry in `all_indices`.
        all_indices: List of `(row, col)` pairs into `candidates`.
        n: Total number of papers.
        top_k: Number of edges to retain per paper.

    Returns:
        Tuple of (top-K candidate indices, top-K scores), both `(N, top_k)`.
    """
    paper_scores: dict[int, list[tuple[int, float]]] = {}
    for idx, (i, j_col) in enumerate(all_indices):
        if i not in paper_scores:
            paper_scores[i] = []
        j = int(candidates[i, j_col])
        paper_scores[i].append((j, float(scores[idx])))

    out_candidates = np.full((n, top_k), -1, dtype=np.int32)
    out_scores = np.zeros((n, top_k), dtype=np.float32)

    for i, cand_scores in paper_scores.items():
        cand_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (j, score) in enumerate(cand_scores[:top_k]):
            out_candidates[i, rank] = j
            out_scores[i, rank] = score

    return out_candidates, out_scores


def _score_facet_candidates(
    facet_name: str,
    candidates_dir: Path,
    output_dir: Path,
    model: LocalReranker,
    facets_map: dict[CorpusId, Facets],
    titles: dict[CorpusId, str],
    top_k: int,
    batch_size: int,
) -> None:
    """Score and prune candidates for a single facet type.

    Args:
        facet_name: One of the canonical facet names.
        candidates_dir: Directory containing `facet_candidates_*.npz` files.
        output_dir: Directory to write `paper_edges_*.npz` files.
        model: Loaded re-ranker model.
        facets_map: Corpus ID to facets mapping.
        titles: Corpus ID to title mapping.
        top_k: Number of edges to retain per paper.
        batch_size: Pairs per forward pass.
    """
    import torch  # noqa: PLC0415

    npz_path = candidates_dir / f"facet_candidates_{facet_name}.npz"
    if not npz_path.exists():
        logger.warning(f"Missing {npz_path}, skipping")
        return

    logger.info(f"=== {facet_name} ===")
    with np.load(npz_path) as npz:
        corpusids: npt.NDArray[np.int64] = npz["corpusids"]
        candidates: npt.NDArray[np.int32] = npz["candidates"]

    all_texts, all_indices = _format_candidate_pairs(
        corpusids, candidates, facets_map, titles
    )

    logger.info(f"  Scoring {len(all_texts)} pairs...")
    with torch.inference_mode():
        scores = model.score_pairs(all_texts, batch_size=batch_size)

    out_candidates, out_scores = _select_top_k(
        candidates, scores, all_indices, len(corpusids), top_k
    )

    out_path = output_dir / f"paper_edges_{facet_name}.npz"
    tmp_path = output_dir / f"paper_edges_{facet_name}.tmp.npz"
    np.savez_compressed(
        tmp_path,
        corpusids=corpusids,
        candidates=out_candidates,
        scores=out_scores,
    )
    tmp_path.rename(out_path)
    logger.info(f"  Saved {out_path}")


def apply_reranker(
    facets_path: Path,
    paper_graphs_path: Path,
    candidates_dir: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    top_k: int = 20,
    batch_size: int = 32,
    device: str = "cuda",
) -> None:
    """Score all candidate pairs with a trained re-ranker and write edge files.

    For each facet type, loads the top-100 candidate pairs, scores them with
    the cross-encoder, keeps the top-K by score, and writes the result as a
    compressed `paper_edges_{name}.npz` file.

    Args:
        facets_path: Path to facets JSONL file.
        paper_graphs_path: Path to paper_graphs JSONL file (for titles).
        candidates_dir: Directory containing `facet_candidates_*.npz` files.
        model_dir: Path to the fine-tuned checkpoint directory.
        output_dir: Directory to write `paper_edges_*.npz` files.
        top_k: Top-K edges to retain per paper per facet.
        batch_size: Pairs per forward pass.
        device: Device for inference (`"cuda"` or `"cpu"`).
    """
    config = RerankerConfig(model_name=str(model_dir), device=device)
    model = LocalReranker.from_pretrained(config)

    facets_map = load_facets_map(facets_path)
    titles_map = load_titles(paper_graphs_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    for facet_name in FACET_NAMES:
        _score_facet_candidates(
            facet_name=facet_name,
            candidates_dir=candidates_dir,
            output_dir=output_dir,
            model=model,
            facets_map=facets_map,
            titles=titles_map,
            top_k=top_k,
            batch_size=batch_size,
        )
