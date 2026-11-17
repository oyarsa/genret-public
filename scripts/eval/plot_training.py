r"""Plot training loss curves and recall@k from checkpoint directories.

Reads ``trainer_state.json`` for loss/eval-loss history.
Optionally reads a retrieval JSONL (from evaluate-codebook or glen evaluate)
and label files to compute recall@k.

Usage:
    # Loss curves only
    uv run python scripts/eval/plot_training.py \\
        --checkpoint output/genret/checkpoints/llama-8b \\
        --label codebook

    # Loss + recall from an existing retrieval JSONL
    uv run python scripts/eval/plot_training.py \\
        --checkpoint output/genret/checkpoints/llama-8b \\
        --retrieval output/genret/eval/llama-8b_500q.jsonl \\
        --label-files output/eval_500/gpt5mini_judgements.jsonl \\
        --label codebook
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import typer


def load_trainer_state(checkpoint_dir: Path) -> Any:
    """Load trainer_state.json from checkpoint or its latest sub-checkpoint."""
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        # Try the most recent checkpoint sub-directory
        subdirs = sorted(
            [d for d in checkpoint_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1]),
        )
        if not subdirs:
            raise FileNotFoundError(f"No trainer_state.json in {checkpoint_dir}")
        state_path = subdirs[-1] / "trainer_state.json"
    return json.loads(state_path.read_text())


def extract_loss_curves(
    state: Any,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (train_steps, train_loss, eval_steps, eval_loss)."""
    train_steps: list[float] = []
    train_loss: list[float] = []
    eval_steps: list[float] = []
    eval_loss: list[float] = []

    for entry in state.get("log_history", []):
        step = entry.get("step", 0)
        if "loss" in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])

    return train_steps, train_loss, eval_steps, eval_loss


def load_retrieval(retrieval_path: Path) -> dict[int, list[int]]:
    """Load retrieval JSONL: {corpusid -> [ranked_corpusids]}.

    Handles both ``query_id``/int-results format and
    ``corpusid``/dict-results format from evaluate-codebook / glen evaluate.
    """
    results: dict[int, list[int]] = {}
    for line in retrieval_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = int(r.get("query_id") or r["corpusid"])
        raw = r.get("results", [])
        if raw and isinstance(raw[0], dict):
            ranked = [int(item["corpusid"]) for item in raw]
        else:
            ranked = [int(x) for x in raw]
        results[qid] = ranked
    return results


def load_labels(label_files: list[Path]) -> dict[int, set[int]]:
    """Load positive labels from judgment JSONL files."""
    positives: dict[int, set[int]] = defaultdict(set)
    for path in label_files:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("related"):
                continue
            positives[int(r["paper_a"])].add(int(r["paper_b"]))
    return positives


def compute_recall(
    retrieval: dict[int, list[int]],
    positives: dict[int, set[int]],
    k_values: list[int],
) -> dict[int, float]:
    """Compute recall@k averaged over queries that have positives."""
    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    for qid, ranked in retrieval.items():
        gold = positives.get(qid, set())
        if not gold:
            continue
        for k in k_values:
            top = set(ranked[:k])
            recalls[k].append(len(gold & top) / len(gold))
    return {k: sum(v) / max(len(v), 1) for k, v in recalls.items()}


def plot_loss(
    ax: plt.Axes,
    train_steps: list[float],
    train_loss: list[float],
    eval_steps: list[float],
    eval_loss: list[float],
    label: str,
) -> None:
    """Plot train and eval loss curves on ax."""
    if train_steps:
        ax.plot(
            train_steps, train_loss, label=f"{label} train", linewidth=1.0, alpha=0.8
        )
    if eval_steps:
        ax.plot(
            eval_steps,
            eval_loss,
            label=f"{label} eval",
            linewidth=1.5,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


def plot_recall(
    ax: plt.Axes,
    recalls: dict[int, float],
    label: str,
) -> None:
    """Bar chart of recall@k values."""
    ks = sorted(recalls)
    vals = [recalls[k] for k in ks]
    x = range(len(ks))
    bars = ax.bar(x, vals, width=0.6, label=label)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"R@{k}" for k in ks])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Recall@k")
    ax.legend()


def main(
    checkpoint: Annotated[
        Path, typer.Option(help="Checkpoint directory containing trainer_state.json.")
    ],
    retrieval: Annotated[
        Path | None,
        typer.Option(help="Retrieval JSONL from evaluate-codebook or glen evaluate."),
    ] = None,
    label_files: Annotated[
        list[Path] | None,
        typer.Option(help="Judgment JSONL files for recall computation."),
    ] = None,
    k_top: Annotated[
        list[int] | None,
        typer.Option(help="k values for recall@k."),
    ] = None,
    label: Annotated[str, typer.Option(help="Label for plot legend.")] = "model",
    output: Annotated[
        Path | None,
        typer.Option(help="Save figure to this path instead of showing interactively."),
    ] = None,
) -> None:
    """Plot training loss curves and optionally recall@k from a checkpoint."""
    if label_files is None:
        label_files = [
            Path("output/eval_500/gpt5mini_judgements.jsonl"),
            Path("output/ablations/gpt5mini_missing_judgements.jsonl"),
        ]
    if k_top is None:
        k_top = [5, 10, 20, 50, 100]

    state = load_trainer_state(checkpoint)
    train_steps, train_loss, eval_steps, eval_loss = extract_loss_curves(state)

    has_retrieval = retrieval is not None and retrieval.exists()
    ncols = 2 if has_retrieval else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    plot_loss(axes[0], train_steps, train_loss, eval_steps, eval_loss, label)

    if has_retrieval:
        retrieved = load_retrieval(retrieval)  # type: ignore[arg-type]
        positives = load_labels(label_files)
        recalls = compute_recall(retrieved, positives, k_top)
        plot_recall(axes[1], recalls, label)
        n_queries = sum(1 for qid in retrieved if positives.get(qid))
        axes[1].set_title(f"Recall@k  (n={n_queries} queries)")

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        typer.echo(f"Saved to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    typer.run(main)
