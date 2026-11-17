"""Rich LLM explanations for retrieval connections, cached on disk.

Extends the template explanations (``explanations.py``) with on-demand
LLM-generated text: given a retrieved paper and its context (the free-text
search query or the recommend seed paper), produce 2-3 grounded sentences
on why the paper is relevant. Each (context, paper, model) pair is
generated once and cached forever, so a public demo pays per distinct
connection, not per view.
"""

import hashlib
import json
import logging
from pathlib import Path

from genret.llm.client import CompletionRequest, OpenAIClient, ReasoningEffort
from genret.paper_parsing import Facets

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You explain why a retrieved research paper is relevant in a literature"
    " exploration tool. Ground the explanation in the provided facets;"
    " point at concrete overlaps or complementary aspects, not generic"
    " similarity. Reply with 2-3 plain sentences. No preamble, no markdown,"
    " no bullet points."
)


def _format_facets(title: str, facets: Facets) -> str:
    lines = [f"Title: {title}"]
    for name, items in facets.to_dict().items():
        if items:
            lines.append(f"{name.capitalize()}:")
            lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def build_prompt(
    target_title: str,
    target_facets: Facets,
    query: str | None,
    seed_title: str | None,
    seed_facets: Facets | None,
) -> str:
    """Build the user prompt for one connection.

    Exactly one of ``query`` or (``seed_title``, ``seed_facets``) provides
    the context the target paper is explained against.
    """
    if query is not None:
        context = f'Search query: "{query}"'
    else:
        assert seed_title is not None
        assert seed_facets is not None
        context = "Query paper the user started from:\n" + _format_facets(
            seed_title, seed_facets
        )

    return (
        f"{context}\n\n"
        f"Retrieved paper:\n{_format_facets(target_title, target_facets)}\n\n"
        "Why is the retrieved paper relevant here?"
    )


async def _call_llm(model: str, user_prompt: str) -> str | None:
    """One LLM completion; patched out in tests and the e2e server."""
    client = OpenAIClient(model=model, reasoning_effort=ReasoningEffort.MINIMAL)
    result = await client.complete_text(
        CompletionRequest(system=SYSTEM_PROMPT, user=user_prompt)
    )
    return result.result


class ExplanationCache:
    """Disk cache of generated explanations, one JSON file per connection."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    @staticmethod
    def key(model: str, corpusid: int, query: str | None, seed: int | None) -> str:
        """Stable cache key for one (model, target, context) connection."""
        raw = json.dumps([model, corpusid, query, seed])
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, key: str) -> str | None:
        """Cached explanation text, or None on miss/corruption."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            text = data["explanation"]
        except (json.JSONDecodeError, KeyError, TypeError):
            path.unlink(missing_ok=True)
            return None
        return text if isinstance(text, str) else None

    def put(self, key: str, explanation: str) -> None:
        """Store an explanation."""
        self._path(key).write_text(
            json.dumps({"explanation": explanation}, indent=2), encoding="utf-8"
        )


async def explain_connection(
    model: str,
    cache: ExplanationCache,
    corpusid: int,
    target_title: str,
    target_facets: Facets,
    query: str | None,
    seed_corpusid: int | None,
    seed_title: str | None,
    seed_facets: Facets | None,
) -> str | None:
    """Generate (or fetch cached) explanation for one retrieval connection.

    Returns:
        The explanation text, or None when the LLM call fails.
    """
    key = ExplanationCache.key(model, corpusid, query, seed_corpusid)
    if cached := cache.get(key):
        logger.info("Explanation cache hit for %d", corpusid)
        return cached

    prompt = build_prompt(target_title, target_facets, query, seed_title, seed_facets)
    text = await _call_llm(model, prompt)
    if text is None:
        logger.error("Explanation generation failed for %d", corpusid)
        return None

    text = text.strip()
    cache.put(key, text)
    return text
