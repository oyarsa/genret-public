"""Estimate extraction token usage, cost, and runtime in two phases.

Phase 1 (`tokenise`) scans source JSONL files, reconstructs prompts/completions,
counts tokens, and writes reusable artefacts.

Phase 2 (`estimate`) reads saved artefacts and computes spend/runtime for chosen
prices and rate limits without re-tokenising.
"""

import argparse
import json
import os
import statistics
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import tiktoken
from tqdm import tqdm


@dataclass(frozen=True)
class DistributionStats:
    """Summary statistics for a token-count distribution."""

    count: int
    total: int
    mean: float
    stddev: float
    minimum: int
    median: float
    p90: int
    p95: int
    maximum: int


@dataclass(frozen=True)
class DatasetStats:
    """Token distributions for sample and target datasets."""

    sample_prompt: DistributionStats
    sample_completion: DistributionStats
    target_prompt: DistributionStats


@dataclass(frozen=True)
class PlanEstimate:
    """Runtime estimate for one rate-limit plan."""

    name: str
    rpm: float
    tpm: float
    minutes_by_requests: float
    minutes_by_tokens: float
    bottleneck: str
    total_minutes: float


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for this estimator script."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate token usage, cost, and runtime for full facet extraction."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["tokenise", "estimate", "all"],
        default="all",
        help="Run tokenisation phase, estimation phase, or both.",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("tmp/extraction_cost_stats"),
        help="Directory for saved token artefacts and summaries.",
    )
    parser.add_argument(
        "--sample-graphs",
        type=Path,
        help="Path to sample paper_graphs.jsonl used in a completed extraction run.",
    )
    parser.add_argument(
        "--sample-facets",
        type=Path,
        help="Path to sample facets.jsonl produced by the same sample run.",
    )
    parser.add_argument(
        "--target-graphs",
        type=Path,
        help="Path to target paper_graphs.jsonl for the full run estimate.",
    )
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=0.15,
        help="Input token price in USD per 1M tokens.",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=0.60,
        help="Output token price in USD per 1M tokens.",
    )
    parser.add_argument(
        "--max-paper-tokens",
        type=int,
        default=100000,
        help="Paper-text truncation limit used when preparing prompts.",
    )
    parser.add_argument(
        "--plan",
        action="append",
        default=[],
        help=(
            "Rate-limit plan as name:rpm,tpm. "
            "Can be passed multiple times, e.g. azure:2.5,250000"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Worker count for parallel tokenisation (default: CPU count).",
    )
    return parser


def _require_paths(args: argparse.Namespace, *, for_tokenise: bool) -> None:
    """Validate required paths for the requested mode."""
    if not for_tokenise:
        return

    missing: list[str] = []
    if args.sample_graphs is None:
        missing.append("--sample-graphs")
    if args.sample_facets is None:
        missing.append("--sample-facets")
    if args.target_graphs is None:
        missing.append("--target-graphs")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required arguments for tokenise mode: {joined}")


def _parse_plan(plan_value: str) -> tuple[str, float, float]:
    """Parse one rate-limit plan argument.

    Args:
        plan_value: Plan string in the format ``name:rpm,tpm``.

    Returns:
        Tuple ``(name, rpm, tpm)``.
    """
    name_part, limits_part = plan_value.split(":", maxsplit=1)
    rpm_str, tpm_str = limits_part.split(",", maxsplit=1)
    return name_part, float(rpm_str), float(tpm_str)


def _encode_len(encoding: tiktoken.Encoding, text: str) -> int:
    """Count tokens while allowing literal special-token-like text."""
    return len(encoding.encode(text, disallowed_special=()))


def _count_non_empty_lines(path: Path) -> int:
    """Count non-empty lines in a UTF-8 text file."""
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _section_to_text_raw(section: dict[str, Any], depth: int = 0) -> str:
    """Convert a raw section dictionary into prompt text."""
    lines: list[str] = []
    indent = "  " * depth
    heading = section.get("heading")
    if isinstance(heading, str) and heading:
        lines.append(f"{indent}## {heading}")

    paragraphs_obj = section.get("paragraphs", [])
    if isinstance(paragraphs_obj, list):
        paragraphs = cast(list[Any], paragraphs_obj)
        for para in paragraphs:
            if isinstance(para, str):
                lines.append(f"{indent}{para}")

    subsections_obj = section.get("subsections", [])
    if isinstance(subsections_obj, list):
        subsections = cast(list[Any], subsections_obj)
        for subsection in subsections:
            if isinstance(subsection, dict):
                subsection_dict = cast(dict[str, Any], subsection)
                lines.append(_section_to_text_raw(subsection_dict, depth + 1))

    return "\n\n".join(lines)


def _paper_to_text_raw(paper: dict[str, Any]) -> str:
    """Convert a raw paper dictionary into prompt text."""
    title = paper.get("title", "")
    parts = [f"# {title}"]

    authors_obj = paper.get("authors", [])
    if isinstance(authors_obj, list) and authors_obj:
        authors = cast(list[Any], authors_obj)
        author_names = [a for a in authors if isinstance(a, str)]
        if author_names:
            parts.append(f"Authors: {', '.join(author_names)}")

    sections_obj = paper.get("sections", [])
    if isinstance(sections_obj, list):
        sections = cast(list[Any], sections_obj)
        for section in sections:
            if isinstance(section, dict):
                section_dict = cast(dict[str, Any], section)
                parts.append(_section_to_text_raw(section_dict))

    return "\n\n".join(parts)


def _make_prompt_raw(
    paper: dict[str, Any],
    encoding: tiktoken.Encoding,
    max_paper_tokens: int,
) -> str:
    """Recreate the extraction prompt from raw JSON dictionaries."""
    paper_text = _paper_to_text_raw(paper)
    paper_tokens = encoding.encode(paper_text, disallowed_special=())
    if len(paper_tokens) > max_paper_tokens:
        paper_text = encoding.decode(paper_tokens[:max_paper_tokens])
        paper_text = f"{paper_text}\n\n[Content truncated]"

    return (
        "Analyse this academic paper and extract the following facets.\n"
        "For each facet, provide 1-5 distinct items (1-2 sentences each).\n\n"
        "1. **Problems**: What problems or research questions does this paper address?\n"
        "2. **Methods**: What approaches, methods, or techniques does the paper propose or use?\n"
        "3. **Results**: What are the main results or findings?\n"
        "4. **Contributions**: What are the contributions or novelties of this work?\n\n"
        "Paper:\n"
        f"{paper_text}\n\n"
        "Respond in JSON format with arrays of strings:\n"
        "{\n"
        '    "problems": ["problem 1", "problem 2", ...],\n'
        '    "methods": ["method 1", "method 2", ...],\n'
        '    "results": ["result 1", "result 2", ...],\n'
        '    "contributions": ["contribution 1", "contribution 2", ...]\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Worker state (initialised once per process via ProcessPoolExecutor)
# ---------------------------------------------------------------------------

_worker_encoding: tiktoken.Encoding
"""Per-worker tiktoken encoding instance."""

_worker_max_paper_tokens: int
"""Per-worker paper token truncation limit."""


def _init_worker(max_paper_tokens: int) -> None:
    """Initialise per-process tiktoken encoding and settings."""
    global _worker_encoding, _worker_max_paper_tokens  # noqa: PLW0603
    _worker_encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    _worker_max_paper_tokens = max_paper_tokens


def _count_prompt_tokens_worker(line: str) -> int:
    """Count prompt tokens for a single paper-graph JSON line."""
    record = json.loads(line)
    if not isinstance(record, dict):
        return 0
    record_dict = cast(dict[str, Any], record)
    prompt = _make_prompt_raw(record_dict, _worker_encoding, _worker_max_paper_tokens)
    return _encode_len(_worker_encoding, prompt)


def _count_completion_tokens_worker(line: str) -> int:
    """Count completion tokens for a single facet JSON line."""
    record = json.loads(line)
    completion_json = json.dumps(
        {
            "problems": record.get("problems", []),
            "methods": record.get("methods", []),
            "results": record.get("results", []),
            "contributions": record.get("contributions", []),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _encode_len(_worker_encoding, completion_json)


# ---------------------------------------------------------------------------
# Parallel tokenisation helpers
# ---------------------------------------------------------------------------


def _non_empty_lines(path: Path) -> Iterator[str]:
    """Yield non-empty stripped lines from a UTF-8 text file."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield stripped


def _tokenize_parallel(
    path: Path,
    worker_fn: Callable[[str], int],
    max_paper_tokens: int,
    workers: int,
    desc: str,
) -> list[int]:
    """Count tokens for every line in a JSONL file, optionally in parallel."""
    total = _count_non_empty_lines(path)
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(max_paper_tokens,),
        ) as pool:
            return list(
                tqdm(
                    pool.map(worker_fn, _non_empty_lines(path), chunksize=64),
                    total=total,
                    desc=desc,
                )
            )
    _init_worker(max_paper_tokens)
    return [
        worker_fn(line) for line in tqdm(_non_empty_lines(path), total=total, desc=desc)
    ]


def _percentile(values: list[int], q: float) -> int:
    """Return a nearest-rank percentile from a non-empty list."""
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * q)
    return sorted_values[index]


def _distribution(values: list[int]) -> DistributionStats:
    """Compute summary statistics for a non-empty integer vector."""
    return DistributionStats(
        count=len(values),
        total=sum(values),
        mean=statistics.mean(values),
        stddev=statistics.pstdev(values),
        minimum=min(values),
        median=statistics.median(values),
        p90=_percentile(values, 0.90),
        p95=_percentile(values, 0.95),
        maximum=max(values),
    )


def _save_tokens(path: Path, values: list[int]) -> None:
    """Write one integer token count per line."""
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def _load_tokens(path: Path) -> list[int]:
    """Load one integer token count per line."""
    with path.open(encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def _fmt_m(value: float) -> str:
    """Format a number in millions with three decimal places."""
    return f"{value / 1_000_000:.3f}M"


def _fmt_money(value: float) -> str:
    """Format a USD amount to two decimal places."""
    return f"${value:,.2f}"


def _minutes_to_human(minutes: float) -> str:
    """Convert minutes to a concise human-readable duration."""
    hours = minutes / 60.0
    days = hours / 24.0
    return f"{minutes:,.1f} min ({hours:,.1f} h, {days:,.2f} d)"


def _estimate_plan(
    name: str,
    rpm: float,
    tpm: float,
    request_count: int,
    total_tokens: int,
) -> PlanEstimate:
    """Estimate minimum runtime under combined RPM/TPM constraints."""
    minutes_by_requests = request_count / rpm
    minutes_by_tokens = total_tokens / tpm
    bottleneck = "requests" if minutes_by_requests >= minutes_by_tokens else "tokens"
    total_minutes = max(minutes_by_requests, minutes_by_tokens)
    return PlanEstimate(
        name=name,
        rpm=rpm,
        tpm=tpm,
        minutes_by_requests=minutes_by_requests,
        minutes_by_tokens=minutes_by_tokens,
        bottleneck=bottleneck,
        total_minutes=total_minutes,
    )


def _tokenise_phase(args: argparse.Namespace) -> DatasetStats:
    """Run phase 1 and persist per-item tokens plus summary statistics."""
    _require_paths(args, for_tokenise=True)
    args.stats_dir.mkdir(parents=True, exist_ok=True)

    sample_graphs = cast(Path, args.sample_graphs)
    sample_facets = cast(Path, args.sample_facets)
    target_graphs = cast(Path, args.target_graphs)
    workers: int = args.workers
    mpt: int = args.max_paper_tokens

    sample_prompt_tokens = _tokenize_parallel(
        sample_graphs,
        _count_prompt_tokens_worker,
        mpt,
        workers,
        "Tokenising sample prompts",
    )
    sample_completion_tokens = _tokenize_parallel(
        sample_facets,
        _count_completion_tokens_worker,
        mpt,
        workers,
        "Tokenising completions",
    )
    target_prompt_tokens = _tokenize_parallel(
        target_graphs,
        _count_prompt_tokens_worker,
        mpt,
        workers,
        "Tokenising target prompts",
    )

    _save_tokens(args.stats_dir / "sample_prompt_tokens.txt", sample_prompt_tokens)
    _save_tokens(
        args.stats_dir / "sample_completion_tokens.txt",
        sample_completion_tokens,
    )
    _save_tokens(args.stats_dir / "target_prompt_tokens.txt", target_prompt_tokens)

    dataset_stats = DatasetStats(
        sample_prompt=_distribution(sample_prompt_tokens),
        sample_completion=_distribution(sample_completion_tokens),
        target_prompt=_distribution(target_prompt_tokens),
    )

    summary_payload: dict[str, Any] = {
        "meta": {
            "sample_graphs": str(sample_graphs),
            "sample_facets": str(sample_facets),
            "target_graphs": str(target_graphs),
            "max_paper_tokens": mpt,
            "workers": workers,
            "tokenizer": "gpt-4o-mini",
        },
        "stats": asdict(dataset_stats),
    }
    with (args.stats_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    return dataset_stats


def _load_dataset_stats(
    stats_dir: Path,
) -> tuple[DatasetStats, list[int], list[int], list[int]]:
    """Load saved token vectors and derive fresh distribution statistics."""
    sample_prompt = _load_tokens(stats_dir / "sample_prompt_tokens.txt")
    sample_completion = _load_tokens(stats_dir / "sample_completion_tokens.txt")
    target_prompt = _load_tokens(stats_dir / "target_prompt_tokens.txt")

    stats = DatasetStats(
        sample_prompt=_distribution(sample_prompt),
        sample_completion=_distribution(sample_completion),
        target_prompt=_distribution(target_prompt),
    )
    return stats, sample_prompt, sample_completion, target_prompt


def _estimate_phase(
    args: argparse.Namespace,
    stats: DatasetStats,
    sample_completion_tokens: list[int],
    target_prompt_tokens: list[int],
) -> None:
    """Run phase 2 from saved token data and print cost/runtime estimates."""
    target_requests = len(target_prompt_tokens)
    target_input_tokens = stats.target_prompt.total

    completion_mean = stats.sample_completion.mean
    completion_p90 = stats.sample_completion.p90

    target_output_tokens_base = round(completion_mean * target_requests)
    target_output_tokens_high = completion_p90 * target_requests

    input_cost = target_input_tokens / 1_000_000 * args.input_price_per_1m
    output_cost_base = target_output_tokens_base / 1_000_000 * args.output_price_per_1m
    output_cost_high = target_output_tokens_high / 1_000_000 * args.output_price_per_1m

    total_tokens_base = target_input_tokens + target_output_tokens_base
    total_tokens_high = target_input_tokens + target_output_tokens_high

    print("=== Dataset and Request Estimate ===")
    print(f"sample_papers={stats.sample_prompt.count}")
    print(f"target_papers={target_requests}")
    print("request_pattern=1 chat.completions call per paper")
    print()

    print("=== Token Estimate ===")
    print(f"sample_prompt_mean={stats.sample_prompt.mean:.2f}")
    print(f"sample_prompt_stddev={stats.sample_prompt.stddev:.2f}")
    print(f"sample_prompt_p90={stats.sample_prompt.p90}")
    print(f"sample_completion_mean={completion_mean:.2f}")
    print(f"sample_completion_stddev={stats.sample_completion.stddev:.2f}")
    print(f"sample_completion_p90={completion_p90}")
    print(f"target_prompt_mean={stats.target_prompt.mean:.2f}")
    print(f"target_prompt_stddev={stats.target_prompt.stddev:.2f}")
    print(f"target_input_tokens={target_input_tokens} ({_fmt_m(target_input_tokens)})")
    print(
        "target_output_tokens_base="
        f"{target_output_tokens_base} ({_fmt_m(target_output_tokens_base)})"
    )
    print(
        "target_output_tokens_high="
        f"{target_output_tokens_high} ({_fmt_m(target_output_tokens_high)})"
    )
    print(f"target_total_tokens_base={total_tokens_base} ({_fmt_m(total_tokens_base)})")
    print(f"target_total_tokens_high={total_tokens_high} ({_fmt_m(total_tokens_high)})")
    print()

    print("=== Cost Estimate ===")
    print(f"input_price_per_1m={args.input_price_per_1m}")
    print(f"output_price_per_1m={args.output_price_per_1m}")
    print(f"input_cost={_fmt_money(input_cost)}")
    print(f"output_cost_base={_fmt_money(output_cost_base)}")
    print(f"output_cost_high={_fmt_money(output_cost_high)}")
    print(f"total_cost_base={_fmt_money(input_cost + output_cost_base)}")
    print(f"total_cost_high={_fmt_money(input_cost + output_cost_high)}")
    print()

    if args.plan:
        print("=== Runtime by Rate-Limit Plan ===")
        for plan_value in args.plan:
            name, rpm, tpm = _parse_plan(plan_value)
            base = _estimate_plan(name, rpm, tpm, target_requests, total_tokens_base)
            high = _estimate_plan(name, rpm, tpm, target_requests, total_tokens_high)
            print(f"plan={name} rpm={rpm} tpm={tpm}")
            print(
                "  base_minutes="
                f"{_minutes_to_human(base.total_minutes)} bottleneck={base.bottleneck}"
            )
            print(
                "  high_minutes="
                f"{_minutes_to_human(high.total_minutes)} bottleneck={high.bottleneck}"
            )

    estimate_payload = {
        "pricing": {
            "input_price_per_1m": args.input_price_per_1m,
            "output_price_per_1m": args.output_price_per_1m,
        },
        "request_count": target_requests,
        "input_tokens": target_input_tokens,
        "output_tokens": {
            "base": target_output_tokens_base,
            "high": target_output_tokens_high,
        },
        "total_tokens": {
            "base": total_tokens_base,
            "high": total_tokens_high,
        },
        "cost_usd": {
            "input": input_cost,
            "output_base": output_cost_base,
            "output_high": output_cost_high,
            "total_base": input_cost + output_cost_base,
            "total_high": input_cost + output_cost_high,
        },
        "sample_completion_count": len(sample_completion_tokens),
    }
    with (args.stats_dir / "estimate.json").open("w", encoding="utf-8") as handle:
        json.dump(estimate_payload, handle, indent=2)


def main() -> None:
    """Run estimator phases based on command-line mode."""
    args = _build_arg_parser().parse_args()

    if args.mode in {"tokenise", "all"}:
        _tokenise_phase(args)

    if args.mode in {"estimate", "all"}:
        stats, _, sample_completion_tokens, target_prompt_tokens = _load_dataset_stats(
            args.stats_dir
        )
        _estimate_phase(args, stats, sample_completion_tokens, target_prompt_tokens)


if __name__ == "__main__":
    main()
