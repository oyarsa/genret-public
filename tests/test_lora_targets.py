"""Tests for LoRA target retargeting on wrapper modules."""

import torch.nn as nn

from genret.training.model import resolve_lora_targets


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _Block(nn.Module):
    def __init__(self, wrapped: bool) -> None:
        super().__init__()
        self.q_proj = _Wrapper() if wrapped else nn.Linear(4, 4)
        self.v_proj = _Wrapper() if wrapped else nn.Linear(4, 4)


class TestResolveLoraTargets:
    """Wrapper modules get .linear-suffixed patterns; plain Linears don't."""

    def test_plain_linear_unchanged(self) -> None:
        """Standard architectures keep their patterns."""
        model = _Block(wrapped=False)
        assert resolve_lora_targets(model, ["q_proj", "v_proj"]) == [
            "q_proj",
            "v_proj",
        ]

    def test_wrapper_retargeted(self) -> None:
        """Gemma-4-style wrappers are retargeted to the inner linear."""
        model = _Block(wrapped=True)
        assert resolve_lora_targets(model, ["q_proj", "v_proj"]) == [
            "q_proj.linear",
            "v_proj.linear",
        ]


class _MixedModel(nn.Module):
    """Gemma-4-like: plain Linear in one tower, wrappers in another."""

    def __init__(self) -> None:
        super().__init__()
        self.language_model = _Block(wrapped=False)
        self.vision_tower = _Block(wrapped=True)


class TestMixedTowers:
    """Mixed matches expand to plain-Linear full names only."""

    def test_mixed_selects_language_tower(self) -> None:
        """Vision-tower wrappers are excluded; language Linears kept."""
        model = _MixedModel()
        out = resolve_lora_targets(model, ["q_proj"])
        assert out == ["language_model.q_proj"]
