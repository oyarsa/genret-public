"""Unit-level smoke tests for the codebook generative-retrieval pipeline.

Exercises the GPU-free pieces: prefix-allowed-tokens construction, RRF
fusion across facet decodes, and the dataset prompt formatting. The
actual ``model.generate`` step is covered by milestone-4 GPU runs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from genret.training.codebook import CodebookParams, build_codebooks
from genret.training.codebook_dataset import (
    SYSTEM_PROMPT_INDEXING,
    SYSTEM_PROMPT_RETRIEVAL,
    _build_messages,
    _target_string,
)
from genret.training.codebook_decoding import (
    FacetDecodeResult,
    build_token_map,
    load_codebook_index,
    make_facet_prefix_fn,
)
from genret.training.codebook_evaluate import _fuse_facet_results
from genret.training.config import (
    FACET_TOKEN_PREFIXES,
    CodebookConfig,
)
from genret.types import CorpusId


def _make_codebook_dir(tmp_path: Path, *, k0: int = 2, k1: int = 2) -> Path:
    """Build a tiny but valid codebook directory on disk."""
    rng = np.random.default_rng(0)
    n_per = 3
    centers = np.array(
        [[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0], [0.0, -5.0]], dtype=np.float32
    )
    vecs = np.concatenate(
        [
            c + rng.normal(scale=0.05, size=(n_per, 2)).astype(np.float32)
            for c in centers
        ]
    )
    corpusids = np.arange(1000, 1000 + vecs.shape[0], dtype=np.int64)

    processed = tmp_path / "processed"
    processed.mkdir()
    for facet in ("problems", "methods", "results", "contributions"):
        np.savez(
            processed / f"facet_graph_{facet}.npz",
            corpusids=corpusids,
            corpus_vectors=vecs.astype(np.float16),
        )

    codebook_dir = tmp_path / "codebooks"
    build_codebooks(
        processed,
        codebook_dir,
        CodebookParams(k0=k0, k1=k1, random_state=0, n_init=2),
    )
    return codebook_dir


def _fake_tokeniser(cfg: CodebookConfig) -> MagicMock:
    """Build a mock tokeniser that resolves each codebook token to a unique id."""
    vocab: dict[str, int] = {
        token: i for i, token in enumerate(cfg.special_tokens(), start=100)
    }
    eos_id = 99
    unk_id = 98

    def convert(token: str) -> int:
        return vocab.get(token, unk_id)

    mock = MagicMock()
    mock.convert_tokens_to_ids.side_effect = convert
    mock.eos_token_id = eos_id
    mock.unk_token_id = unk_id
    return mock


def test_build_token_map_resolves_every_codebook_token(tmp_path: Path) -> None:
    """Every special token registered by the config must come back from the map."""
    codebook_dir = _make_codebook_dir(tmp_path, k0=2, k1=2)
    cfg = CodebookConfig(codebook_dir=codebook_dir, k0=2, k1=2)
    tokens = build_token_map(cfg, _fake_tokeniser(cfg))

    # 2 delimiters + 4 facets * (2 L0 + 2 L1) = 18 distinct ids.
    assert tokens.code_start_id != tokens.code_end_id
    assert len(tokens.l0_id_to_value) == 4 * 2
    assert len(tokens.l1_id_to_value) == 4 * 2
    for facet in FACET_TOKEN_PREFIXES:
        assert sorted(tokens.l0_ids[facet]) == [0, 1]
        assert sorted(tokens.l1_ids[facet]) == [0, 1]


def test_prefix_fn_walks_four_steps_through_codebook(tmp_path: Path) -> None:
    """Step-by-step prefix constraints follow the codebook structure.

    Step 0 emits CODE_START; step 1 allows only populated L0; step 2
    allows only L1 children of the chosen L0; step 3 emits CODE_END.
    """
    codebook_dir = _make_codebook_dir(tmp_path, k0=2, k1=2)
    cfg = CodebookConfig(codebook_dir=codebook_dir, k0=2, k1=2)
    index = load_codebook_index(codebook_dir)
    tokens = build_token_map(cfg, _fake_tokeniser(cfg))

    prompt_len = 5
    prefix_fn = make_facet_prefix_fn(
        "problems", index=index, tokens=tokens, prompt_len=prompt_len
    )

    # Build a fake beam_input_ids tensor (prompt then generated tokens)
    def at(step_tokens: list[int]) -> torch.Tensor:
        prefix = torch.zeros(prompt_len, dtype=torch.long)
        return torch.cat([prefix, torch.tensor(step_tokens, dtype=torch.long)])

    # Step 0: only CODE_START allowed
    assert prefix_fn(0, at([])) == [tokens.code_start_id]

    # Step 1: only populated L0 tokens for "problems"
    populated_l0 = sorted(index.populated_l1_by_l0["problems"].keys())
    expected_l0 = [tokens.l0_ids["problems"][v] for v in populated_l0]
    actual_l0 = prefix_fn(0, at([tokens.code_start_id]))
    assert sorted(actual_l0) == sorted(expected_l0)

    # Step 2: only populated L1 under the chosen L0
    chosen_l0 = populated_l0[0]
    chosen_l0_id = tokens.l0_ids["problems"][chosen_l0]
    populated_l1 = index.populated_l1_by_l0["problems"][chosen_l0]
    expected_l1 = [tokens.l1_ids["problems"][v] for v in populated_l1]
    actual_l1 = prefix_fn(0, at([tokens.code_start_id, chosen_l0_id]))
    assert sorted(actual_l1) == sorted(expected_l1)

    # Step 3: only CODE_END
    chosen_l1_id = tokens.l1_ids["problems"][populated_l1[0]]
    assert prefix_fn(0, at([tokens.code_start_id, chosen_l0_id, chosen_l1_id])) == [
        tokens.code_end_id
    ]

    # Step 4+: EOS
    end_id = tokens.code_end_id
    assert prefix_fn(
        0,
        at([tokens.code_start_id, chosen_l0_id, chosen_l1_id, end_id]),
    ) == [tokens.eos_id]


def test_fuse_rrf_combines_facet_rankings() -> None:
    """RRF fusion ranks papers by reciprocal rank sum across facets."""
    paper_a = CorpusId(1)
    paper_b = CorpusId(2)
    paper_c = CorpusId(3)
    per_facet = {
        "problems": [
            FacetDecodeResult(
                facet="problems",
                rank=0,
                score=1.0,
                l0=0,
                l1=0,
                paper_ids=[paper_a],
            ),
            FacetDecodeResult(
                facet="problems",
                rank=1,
                score=0.9,
                l0=0,
                l1=1,
                paper_ids=[paper_b],
            ),
        ],
        "methods": [
            FacetDecodeResult(
                facet="methods",
                rank=0,
                score=1.0,
                l0=0,
                l1=0,
                paper_ids=[paper_b],
            ),
        ],
        "results": [],
        "contributions": [
            FacetDecodeResult(
                facet="contributions",
                rank=0,
                score=1.0,
                l0=0,
                l1=0,
                paper_ids=[paper_c],
            ),
        ],
    }
    fused = _fuse_facet_results(per_facet)
    fused_ids = [r.corpusid for r in fused]
    # paper_b should be top because it appears in two facets at high rank.
    assert fused_ids[0] == paper_b
    # paper_a and paper_c each appear once at rank 0 — their RRF scores tie.
    assert set(fused_ids[1:]) == {paper_a, paper_c}
    # paper_b's facet_ranks should record both facets it surfaced through.
    pb = next(r for r in fused if r.corpusid == paper_b)
    assert pb.facets == ["methods", "problems"]
    assert pb.facet_ranks == {"problems": 1, "methods": 0}


def test_dataset_prompts_use_distinct_indexing_and_retrieval_systems() -> None:
    """Indexing and retrieval examples should use different system prompts."""
    indexing_msgs = _build_messages("problems", "- foo bar", kind="indexing")
    retrieval_msgs = _build_messages("problems", "- foo bar", kind="retrieval")
    assert indexing_msgs[0]["role"] == "system"
    assert (
        SYSTEM_PROMPT_INDEXING.format(facet_type="Problems")
        == indexing_msgs[0]["content"]
    )
    assert (
        SYSTEM_PROMPT_RETRIEVAL.format(facet_type="Problems")
        == retrieval_msgs[0]["content"]
    )
    assert indexing_msgs[1]["content"].startswith("Problems:")


def test_target_string_uses_codebook_special_tokens() -> None:
    """The target should be `<CODE_START> <X_L0_a> <X_L1_b> <CODE_END>`."""
    cfg = CodebookConfig(codebook_dir=Path("./_unused"), k0=2, k1=2)
    target = _target_string(cfg, "methods", (1, 0))
    assert target == "<CODE_START> <M_L0_1> <M_L1_0> <CODE_END>"


def test_codebook_index_serialises_round_trip(tmp_path: Path) -> None:
    """`load_codebook_index` reconstructs the lookup deterministically."""
    codebook_dir = _make_codebook_dir(tmp_path, k0=2, k1=2)
    index = load_codebook_index(codebook_dir)

    # Every populated (L0, L1) leaf is in code_to_papers.
    raw = json.loads((codebook_dir / "code_to_papers.json").read_text())
    for facet, leaves in raw.items():
        for key, papers in leaves.items():
            l0_str, l1_str = key.split(",")
            l0, l1 = int(l0_str), int(l1_str)
            decoded = index.code_to_papers[facet].get((l0, l1), [])
            assert [int(c) for c in decoded] == [int(c) for c in papers]
        # populated_l1_by_l0 only contains non-empty (L0, L1) cells.
        populated = index.populated_l1_by_l0[facet]
        for l0, l1_list in populated.items():
            for l1 in l1_list:
                assert index.code_to_papers[facet][(l0, l1)], (
                    "populated_l1_by_l0 must only list non-empty leaves"
                )


# Make ruff happy about the unused rng import in the future.
del random
