"""Local facet extraction with a directly-loaded HuggingFace model.

Mirrors the API-based ``extract`` pipeline but runs a local causal LM
(e.g. ``Qwen/Qwen3-32B``) via transformers instead of the OpenAI provider.

The model is prompted for a simple labelled-bullet format (one block per
facet) rather than strict JSON — local models follow that far more reliably,
and the downstream consumers only need the four facet string lists. Output is
written to the same ``facets.jsonl.zst`` schema as the API path, so the
existing ``evaluate-facets`` tooling works unchanged.
"""

# pyright: basic

import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from genret._shed.torch import generate as model_generate
from genret._shed.transformers import apply_chat_template_str
from genret.llm.models import write_run_metadata
from genret.paper_parsing import Facets, PaperGraph
from genret.processing.facet_extract_cmd import (
    SectionMode,
    _prepare_prompt_paper_text,  # pyright: ignore[reportPrivateUsage]
    append_processed_id,
    iter_paper_graphs,
    paper_to_text,
    prepare_extract_state,
    select_core_section_text,
)
from genret.processing.local_judge import load_model
from genret.types import CorpusId
from genret.utils import die, open_auto, resolve_jsonl, track_progress

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are an expert at analysing academic papers. Extract concise, specific "
    "facets and follow the requested output format exactly."
)

_USER_TEMPLATE = """\
Analyse this academic paper and extract four facets. For each facet, list 1-5 \
distinct items. Each item is a concise 1-2 sentence statement on its own line \
starting with "- ".

Respond using EXACTLY this format, with these four uppercase headers and nothing \
else (no preamble, no commentary):

PROBLEMS:
- <problem or research question the paper addresses>
METHODS:
- <approach, method, or technique the paper proposes or uses>
RESULTS:
- <main result or finding>
CONTRIBUTIONS:
- <contribution or novelty of this work>

Paper:
{paper_text}"""

_FACET_NAMES = ("problems", "methods", "results", "contributions")

# Header line such as "PROBLEMS:", "**Methods**", "## Results", "3. Contributions".
_HEADER_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:\d+[.)]\s*)?\**\s*"
    r"(problems?|methods?|results?|contributions?)"
    r"\s*\**\s*:?\s*$",
    re.IGNORECASE,
)

# Leading bullet/number markers to strip from item lines.
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _canonical_facet(raw: str) -> str:
    """Map a matched header word to its canonical (plural) facet key."""
    low = raw.lower().rstrip("s")
    match low:
        case "problem":
            return "problems"
        case "method":
            return "methods"
        case "result":
            return "results"
        case "contribution":
            return "contributions"
        case _:
            return ""


def parse_facets_text(text: str, *, max_items: int = 5) -> dict[str, list[str]]:
    """Parse a labelled-bullet generation into per-facet item lists.

    Forgiving parser: detects the four facet headers in any common markdown
    decoration and collects the bullet/plain lines under each until the next
    header. Items are de-duplicated (case-insensitive) and capped at
    *max_items*.

    Args:
        text: Raw model generation.
        max_items: Maximum items kept per facet.

    Returns:
        Mapping of facet name to its list of item strings (possibly empty).
    """
    out: dict[str, list[str]] = {name: [] for name in _FACET_NAMES}
    seen: dict[str, set[str]] = {name: set() for name in _FACET_NAMES}
    current: str | None = None

    for line in text.splitlines():
        if header := _HEADER_RE.match(line):
            current = _canonical_facet(header.group(1))
            continue
        if current is None:
            continue

        item = _BULLET_RE.sub("", line).strip()
        # Drop stray markdown emphasis around the whole line.
        item = item.strip("*` ").strip()
        if not item:
            continue

        key = item.lower()
        if key in seen[current]:
            continue
        if len(out[current]) >= max_items:
            continue
        seen[current].add(key)
        out[current].append(item)

    return out


def build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    paper: PaperGraph,
    *,
    max_input_tokens: int,
    section_mode: SectionMode = SectionMode.CORE,
) -> str:
    """Render the chat-templated extraction prompt for one paper."""
    paper_text = _prepare_prompt_paper_text(
        paper, max_tokens=max_input_tokens, section_mode=section_mode
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(paper_text=paper_text)},
    ]
    return apply_chat_template_str(tokenizer, messages, enable_thinking=False)


@torch.inference_mode()
def _generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
) -> list[str]:
    """Greedily generate completions for a batch of rendered prompts."""
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    ).to(model.device)

    generated = model_generate(
        model,
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Left-padded input → uniform prompt length; slice off the prompt tokens.
    new_tokens = generated[:, encoded["input_ids"].shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def _batched(items: list[PaperGraph], size: int) -> Iterator[list[PaperGraph]]:
    """Yield successive *size*-length chunks from *items*."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_local_extract(
    splits_dir: Path,
    filter_ids: set[CorpusId],
    output_dir: Path,
    *,
    model_id: str = "Qwen/Qwen3-32B",
    batch_size: int = 8,
    max_input_tokens: int = 16000,
    max_new_tokens: int = 768,
    section_mode: SectionMode = SectionMode.CORE,
    quantize_4bit: bool = False,
    force: bool = False,
    limit: int = 0,
) -> None:
    """Extract facets for a splits subset using a local HuggingFace model.

    Args:
        splits_dir: Splits directory containing ``processed/paper_graphs.jsonl``.
        filter_ids: Restrict extraction to these corpus IDs (empty = all).
        output_dir: Output directory for ``facets.jsonl.zst`` and sidecars.
        model_id: HuggingFace model ID to load.
        batch_size: Papers generated per forward batch.
        max_input_tokens: Truncate paper text to this many tokens.
        max_new_tokens: Generation budget per paper.
        section_mode: Whole paper (``full``) or core sections only (``core``).
        quantize_4bit: Load the model in 4-bit NF4.
        force: Reprocess all papers (backs up previous output).
        limit: Maximum papers to process (0 = all).
    """
    input_dir = splits_dir / "processed"
    graphs_path = resolve_jsonl(input_dir / "paper_graphs.jsonl.gz")
    if not graphs_path.exists():
        die(f"Paper graphs not found: {graphs_path}\nRun 'parse' first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    facets_path = output_dir / "facets.jsonl.zst"
    progress_path = output_dir / "facets_progress.txt"

    processed_ids, facets_mode = prepare_extract_state(
        progress_path, facets_path, force
    )
    if processed_ids:
        logger.info("Resuming: %d papers already processed", len(processed_ids))

    papers = list(
        iter_paper_graphs(graphs_path, processed_ids, filter_ids, limit=limit)
    )
    if not papers:
        logger.info("All papers already processed!")
        return

    # Sort by selected-text length so each batch holds similar-length inputs:
    # less padding waste, steadier memory (fewer OOM spikes from one long paper
    # padding a whole batch), better throughput. Output is keyed by corpusid,
    # so processing order does not affect results.
    def _approx_len(paper: PaperGraph) -> int:
        if section_mode is SectionMode.CORE:
            return len(select_core_section_text(paper))
        return len(paper_to_text(paper))

    papers.sort(key=_approx_len)

    logger.info("Papers to process: %d", len(papers))
    logger.info("Model: %s (%s)", model_id, "4-bit" if quantize_4bit else "float16")
    logger.info(
        "Batch size: %d | max input tokens: %d | sections: %s",
        batch_size,
        max_input_tokens,
        section_mode.value,
    )

    model, tokenizer = load_model(model_id, quantize_4bit=quantize_4bit)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    n_batches = (len(papers) + batch_size - 1) // batch_size
    with open_auto(facets_path, facets_mode) as writer:
        for batch in track_progress(
            _batched(papers, batch_size),
            desc="Extracting facets (local)...",
            total=n_batches,
        ):
            prompts = [
                build_prompt(
                    tokenizer,
                    p,
                    max_input_tokens=max_input_tokens,
                    section_mode=section_mode,
                )
                for p in batch
            ]
            completions = _generate_batch(
                model, tokenizer, prompts, max_new_tokens=max_new_tokens
            )
            for paper, text in zip(batch, completions, strict=True):
                parsed = parse_facets_text(text)
                facets = Facets(
                    corpusid=paper.corpusid,
                    problems=parsed["problems"],
                    methods=parsed["methods"],
                    results=parsed["results"],
                    contributions=parsed["contributions"],
                )
                writer.write(facets.model_dump_json() + "\n")
                writer.flush()
                append_processed_id(progress_path, paper.corpusid)

    write_run_metadata(
        output_dir / "facets_run.json",
        model=model_id,
        model_short=model_id.split("/")[-1],
        temperature=0.0,
        seed=0,
        total_cost_usd=0.0,
        reasoning_effort="none",
    )
    logger.info("Local facet extraction complete: %s", facets_path)
