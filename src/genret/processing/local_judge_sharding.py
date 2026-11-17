"""Sharding helpers for multi-GPU local judge runs."""

from pathlib import Path

from genret.processing.llm_judge import PairToJudge, ScoredPair
from genret.utils import open_auto, resolve_jsonl


def _validate_shard_args(*, shard_index: int, num_shards: int) -> None:
    """Validate a 0-based shard selection.

    `shard_index` must be non-negative, `num_shards` must be positive and
    `shard_index` <= `num_shards`.

    Args:
        shard_index: Selected shard index.
        num_shards: Total number of shards.

    Raises:
        ValueError: If the shard configuration is invalid.
    """
    if num_shards <= 0:
        raise ValueError("`num_shards` must be positive.")
    if shard_index < 0:
        raise ValueError("`shard_index` must be non-negative.")
    if shard_index >= num_shards:
        raise ValueError("`shard_index` must be smaller than `num_shards`.")


def select_pairs_for_shard(
    pairs: list[PairToJudge],
    *,
    shard_index: int,
    num_shards: int,
) -> list[PairToJudge]:
    """Select one stable round-robin shard from a pair list.

    Args:
        pairs: Full ordered pair list.
        shard_index: 0-based shard index.
        num_shards: Total number of shards.

    Returns:
        The subset assigned to this shard.
    """
    _validate_shard_args(shard_index=shard_index, num_shards=num_shards)
    return [
        pair
        for position, pair in enumerate(pairs)
        if position % num_shards == shard_index
    ]


def merge_scored_shards(
    *,
    input_paths: list[Path],
    output_path: Path,
) -> dict[str, str | int]:
    """Merge multiple scored-shard JSONL files into one output file.

    Args:
        input_paths: Shard output files to merge.
        output_path: Destination JSONL file.

    Returns:
        Summary metadata about the merged output.

    Raises:
        ValueError: If shard outputs disagree on mode/range or contain duplicate
            pair keys.
    """
    if not input_paths:
        raise ValueError("Need at least one scored shard to merge.")

    resolved_inputs = [resolve_jsonl(path) for path in input_paths]
    resolved_output = resolve_jsonl(output_path)

    merged_pairs: list[ScoredPair] = []
    seen_keys: set[tuple[int, int, str]] = set()
    merged_mode: str | None = None
    merged_score_range: str | None = None

    for input_path in resolved_inputs:
        with open_auto(input_path, "r") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if not stripped:
                    continue

                pair = ScoredPair.model_validate_json(stripped)
                pair_key = (pair.paper_a, pair.paper_b, pair.source)
                if pair_key in seen_keys:
                    raise ValueError(f"Duplicate pair in shard inputs: {pair_key}")

                if merged_mode is None:
                    merged_mode = pair.mode.value
                elif pair.mode.value != merged_mode:
                    raise ValueError(
                        "Shard outputs use different scoring modes: "
                        f"{merged_mode!r} vs {pair.mode.value!r}"
                    )

                if merged_score_range is None:
                    merged_score_range = pair.score_range.value
                elif pair.score_range.value != merged_score_range:
                    raise ValueError(
                        "Shard outputs use different score ranges: "
                        f"{merged_score_range!r} vs {pair.score_range.value!r}"
                    )

                seen_keys.add(pair_key)
                merged_pairs.append(pair)

    merged_pairs.sort(key=lambda pair: (pair.paper_a, pair.paper_b, pair.source))

    with open_auto(resolved_output, "w") as file_handle:
        for pair in merged_pairs:
            file_handle.write(pair.model_dump_json())
            file_handle.write("\n")

    return {
        "output_path": str(resolved_output),
        "pair_count": len(merged_pairs),
        "mode": merged_mode or "",
        "score_range": merged_score_range or "",
    }
