"""Model registry: synonyms, costs, rate limits, and run metadata.

Centralises all per-model configuration — dated snapshot pinning, token pricing,
and provider rate limits — so that `client.py` can remain focused on the
completion logic.
"""

import json
from pathlib import Path

MODEL_SYNONYMS: dict[str, str] = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "gpt-5-nano": "gpt-5-nano-2025-08-07",
    "gpt-5": "gpt-5-2025-08-07",
    "gpt-5.2": "gpt-5.2-2025-12-11",
    "gpt-5.4": "gpt-5.4-2026-03-05",
    "gpt-5.4-nano": "gpt-5.4-nano-2026-03-17",
    "gpt-5.4-mini": "gpt-5.4-mini-2026-03-17",
    # No dated snapshot exists for Luna; the bare name is the pinned id
    "gpt-5.6-luna": "gpt-5.6-luna",
}
"""Map of short model aliases to dated snapshot versions.

Ensures reproducible outputs by pinning to a specific model snapshot rather than
relying on OpenAI's rolling alias resolution.
"""

MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4.1-nano-2025-04-14": (0.10, 0.40),
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "gpt-4.1-2025-04-14": (2.00, 8.00),
    "gpt-5-mini-2025-08-07": (0.25, 2.00),
    "gpt-5-nano-2025-08-07": (0.05, 0.40),
    "gpt-5-2025-08-07": (1.75, 14.00),
    "gpt-5.2-2025-12-11": (1.75, 14.00),
    "gpt-5.4-2026-03-05": (2.50, 15.00),
    "gpt-5.4-nano-2026-03-17": (0.20, 1.25),
    "gpt-5.4-mini-2026-03-17": (0.75, 4.50),
    "gpt-5.6-luna": (0.20, 1.20),
}
"""Per-model token pricing as (input_cost_per_1m, output_cost_per_1m) in USD.

Costs are for the dated snapshot versions. Unknown models are treated as zero-cost
to avoid blocking execution.
"""

CHAT_RATE_LIMITS: dict[str, dict[str, tuple[int, int]]] = {
    "openai": {
        "gpt-4o-mini": (30_000, 150_000_000),
        "gpt-4o": (10_000, 30_000_000),
        "gpt-4.1-nano": (30_000, 150_000_000),
        "gpt-4.1-mini": (30_000, 150_000_000),
        "gpt-4.1": (10_000, 30_000_000),
        "gpt-5-mini": (30_000, 180_000_000),
        "gpt-5-nano": (30_000, 180_000_000),
        "gpt-5": (15_000, 40_000_000),
        "gpt-5.2": (15_000, 40_000_000),
        "gpt-5.4-nano": (30_000, 180_000_000),
        "gpt-5.4-mini": (30_000, 180_000_000),
        "gpt-5.4": (15_000, 40_000_000),
        "gpt-5.6-luna": (30_000, 180_000_000),
    },
    "azure-openai": {
        "gpt-5-nano": (3_314, 3_314_000),
        "gpt-4.1-mini": (1_105, 1_105_000),
        "gpt-5-mini": (250, 250_000),
        "gpt-5.4": (250, 250_000),
        "gpt-4.1": (250, 250_000),
        "gpt-4o-mini": (2_500, 250_000),
        "gpt-4o": (1_800, 300_000),
    },
}
"""Per-provider, per-model rate limits as (requests/min, tokens/min) tuples.

OpenAI defaults based on Tier-5 limits. Unknown models fall back to
`DEFAULT_RATE_LIMIT`.
"""

DEFAULT_RATE_LIMIT: tuple[int, int] = (500, 200_000)
"""Fallback rate limit for models not listed in CHAT_RATE_LIMITS."""

FIXED_TEMPERATURE_PREFIXES: tuple[str, ...] = ("gpt-5",)
"""Model prefixes that do not accept a temperature parameter."""


def resolve_model(model: str) -> str:
    """Resolve a short model alias to its dated snapshot version.

    If *model* is already a dated version or not in the synonym registry,
    it is returned unchanged.

    Args:
        model: Short model name (e.g. `"gpt-4o-mini"`) or dated version.

    Returns:
        The dated snapshot version if a synonym exists, otherwise *model* unchanged.
    """
    return MODEL_SYNONYMS.get(model, model)


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute the estimated API cost for a single completion call.

    Accepts both short aliases (e.g. `"gpt-5-mini"`) and dated snapshot names
    (e.g. `"gpt-5-mini-2025-08-07"`).  Returns `0.0` for unknown models.

    Args:
        model: Model name — short alias or dated snapshot.
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.

    Returns:
        Estimated cost in USD. 0 for unknown models.
    """
    resolved = resolve_model(model)
    costs = MODEL_COSTS.get(resolved)
    if costs is None:
        return 0.0

    input_per_1m, output_per_1m = costs
    return (
        prompt_tokens * input_per_1m + completion_tokens * output_per_1m
    ) / 1_000_000


def write_run_metadata(
    path: Path,
    *,
    model: str,
    model_short: str,
    temperature: float,
    seed: int,
    total_cost_usd: float,
    reasoning_effort: str,
) -> None:
    """Write a pretty-printed JSON sidecar file with run metadata.

    Args:
        path: Output file path (e.g. `facets_run.json`).
        model: Resolved dated model name.
        model_short: Original short model alias.
        temperature: Temperature used for completions.
        seed: Seed used for reproducibility.
        total_cost_usd: Accumulated total cost in USD.
        reasoning_effort: Reasoning effort level used for the run.
    """
    metadata = {
        "model": model,
        "model_short": model_short,
        "temperature": temperature,
        "seed": seed,
        "reasoning_effort": reasoning_effort,
        "total_cost_usd": round(total_cost_usd, 6),
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
