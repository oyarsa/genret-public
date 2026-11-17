"""Prompt variants and facet formatting for the local judge."""

from dataclasses import dataclass
from enum import StrEnum

from genret.paper_parsing import Facets


@dataclass(frozen=True, kw_only=True)
class _FacetFormatConfig:
    """Facet-formatting settings for one local-judge prompt variant."""

    max_items_per_section: int
    include_results: bool
    max_chars_per_item: int | None = None


class PromptVariant(StrEnum):
    """Prompt-compression variants supported by the local judge."""

    BASELINE_COMPACT = "baseline_compact"
    TOP2_PER_SECTION = "top2_per_section"
    TOP2_NO_RESULTS = "top2_no_results"
    TOP1_PER_SECTION = "top1_per_section"
    TOP1_NO_RESULTS = "top1_no_results"
    TOP2_TRUNCATED = "top2_truncated"

    def to_config(self) -> _FacetFormatConfig:
        """Return the facet-formatting configuration for one prompt variant.

        Args:
            variant: Prompt variant to configure.

        Returns:
            Formatting configuration for that variant.
        """
        match self:
            case PromptVariant.TOP2_PER_SECTION:
                return _FacetFormatConfig(max_items_per_section=2, include_results=True)
            case PromptVariant.TOP2_NO_RESULTS:
                return _FacetFormatConfig(
                    max_items_per_section=2, include_results=False
                )
            case PromptVariant.TOP1_PER_SECTION:
                return _FacetFormatConfig(max_items_per_section=1, include_results=True)
            case PromptVariant.TOP1_NO_RESULTS:
                return _FacetFormatConfig(
                    max_items_per_section=1, include_results=False
                )
            case PromptVariant.TOP2_TRUNCATED:
                return _FacetFormatConfig(
                    max_items_per_section=2,
                    include_results=True,
                    max_chars_per_item=160,
                )
            case PromptVariant.BASELINE_COMPACT:
                return _FacetFormatConfig(max_items_per_section=5, include_results=True)


def _truncate_item(text: str, *, max_chars: int | None) -> str:
    """Truncate one facet bullet for compact prompting.

    Args:
        text: Original facet bullet.
        max_chars: Maximum characters to keep, or None for no truncation.

    Returns:
        Original or truncated facet text.
    """
    stripped = text.strip()
    if max_chars is None or len(stripped) <= max_chars:
        return stripped

    truncated = stripped[: max_chars - 1].rstrip()
    return f"{truncated}…"


def format_facets_for_local_judge(facets: Facets, *, variant: PromptVariant) -> str:
    """Format a paper's facets for the local judge prompt.

    This formatter always omits `Corpus ID`, then applies a prompt-compression
    variant by selecting a limited number of bullets per section, optionally
    removing the `Results` section, and optionally truncating long bullets.

    Args:
        facets: Extracted facets for one paper.
        variant: Prompt-compression variant to apply.

    Returns:
        Human-readable facet summary for local judging.
    """
    config = variant.to_config()

    sections = [
        ("Problems", tuple(facets.problems)),
        ("Methods", tuple(facets.methods)),
    ]
    if config.include_results:
        sections.append(("Results", tuple(facets.results)))
    sections.append(("Contributions", tuple(facets.contributions)))

    parts: list[str] = []
    for name, items in sections:
        selected_items = items[: config.max_items_per_section]
        if not selected_items:
            continue

        bullet_list = "\n".join(
            f"- {_truncate_item(item, max_chars=config.max_chars_per_item)}"
            for item in selected_items
        )
        parts.append(f"**{name}:**\n{bullet_list}")

    return "\n\n".join(parts)
