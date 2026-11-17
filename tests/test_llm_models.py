"""Unit tests for the llm.models module."""

import json
from pathlib import Path

from conftest import approx
from genret.llm.models import (
    MODEL_COSTS,
    MODEL_SYNONYMS,
    compute_cost,
    resolve_model,
    write_run_metadata,
)


class TestResolveModel:
    """Tests for the resolve_model function."""

    def test_resolves_known_synonym(self) -> None:
        """Should map short alias to dated snapshot."""
        assert resolve_model("gpt-4o-mini") == "gpt-4o-mini-2024-07-18"

    def test_resolves_all_synonyms(self) -> None:
        """Every synonym should resolve to a different (dated) string."""
        for short, dated in MODEL_SYNONYMS.items():
            assert resolve_model(short) == dated

    def test_passes_through_dated_version(self) -> None:
        """Should return dated versions unchanged."""
        assert resolve_model("gpt-4o-mini-2024-07-18") == "gpt-4o-mini-2024-07-18"

    def test_passes_through_unknown_model(self) -> None:
        """Should return unknown model names unchanged."""
        assert resolve_model("some-future-model") == "some-future-model"


class TestComputeCost:
    """Tests for the compute_cost function."""

    def test_known_model_cost(self) -> None:
        """Should compute correct cost for a known model."""
        # gpt-4o-mini-2024-07-18: $0.15/1M input, $0.60/1M output
        cost = compute_cost("gpt-4o-mini-2024-07-18", 1_000_000, 1_000_000)
        assert cost == approx(0.75)

    def test_small_token_counts(self) -> None:
        """Should handle small token counts correctly."""
        cost = compute_cost("gpt-4o-mini-2024-07-18", 1000, 500)
        expected = (1000 * 0.15 + 500 * 0.60) / 1_000_000
        assert cost == approx(expected)

    def test_unknown_model_returns_zero(self) -> None:
        """Should return 0.0 for models not in the cost registry."""
        assert compute_cost("unknown-model", 10000, 5000) == 0.0

    def test_zero_tokens(self) -> None:
        """Should return 0.0 when both token counts are zero."""
        assert compute_cost("gpt-4o-mini-2024-07-18", 0, 0) == 0.0

    def test_all_models_have_costs(self) -> None:
        """Every synonym target should have a cost entry."""
        for dated_model in MODEL_SYNONYMS.values():
            assert dated_model in MODEL_COSTS, f"Missing cost entry for {dated_model}"


class TestWriteRunMetadata:
    """Tests for the write_run_metadata function."""

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        """Should write a valid, pretty-printed JSON file."""
        output = tmp_path / "run.json"
        write_run_metadata(
            output,
            model="gpt-4o-mini-2024-07-18",
            model_short="gpt-4o-mini",
            temperature=0.0,
            seed=0,
            total_cost_usd=0.123456,
            reasoning_effort="default",
        )

        text = output.read_text(encoding="utf-8")
        assert text.endswith("\n")

        data = json.loads(text)
        assert data["model"] == "gpt-4o-mini-2024-07-18"
        assert data["model_short"] == "gpt-4o-mini"
        assert data["temperature"] == 0.0
        assert data["seed"] == 0
        assert data["total_cost_usd"] == 0.123456
        assert data["reasoning_effort"] == "default"

    def test_pretty_printed(self, tmp_path: Path) -> None:
        """Should use indented formatting."""
        output = tmp_path / "run.json"
        write_run_metadata(
            output,
            model="gpt-4o-mini-2024-07-18",
            model_short="gpt-4o-mini",
            temperature=0.0,
            seed=42,
            total_cost_usd=0.0,
            reasoning_effort="default",
        )

        text = output.read_text(encoding="utf-8")
        # Pretty-printed JSON has newlines between fields
        assert "\n" in text.rstrip("\n")

    def test_rounds_cost(self, tmp_path: Path) -> None:
        """Should round cost to 6 decimal places."""
        output = tmp_path / "run.json"
        write_run_metadata(
            output,
            model="m",
            model_short="m",
            temperature=0.0,
            seed=0,
            total_cost_usd=0.1234567890,
            reasoning_effort="default",
        )

        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["total_cost_usd"] == 0.123457
