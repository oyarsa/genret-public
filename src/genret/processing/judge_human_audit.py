"""Interactive human audit workflow for comparing judge outputs.

Shows two papers side by side, asks the user for a related/unrelated label,
reveals model predictions, and records agreement statistics.
"""

import json
import random
import sys
import termios
import textwrap
import tty
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from genret.paper_parsing import Facets, load_facets_map
from genret.processing.llm_judge import JudgedPair, load_judged_pairs
from genret.types import CorpusId
from genret.utils import die

_console = Console()


def _colour_related_label(related: bool) -> str:
    """Render a related/unrelated label with audit colours.

    Args:
        related: Whether the label is related.

    Returns:
        Rich-markup label string.
    """
    if related:
        return "[green]related[/green]"
    return "[red]unrelated[/red]"


class ModelPrediction(BaseModel, frozen=True):
    """Single model prediction shown in the audit UI."""

    related: bool
    """Whether the model judged the pair related."""
    confidence: float
    """Model confidence for the judgement."""
    relationship_type: str
    """Relationship type label emitted by the model."""
    reasoning: str
    """Model explanation shown after the human annotation."""


class HumanAuditRecord(BaseModel, frozen=True):
    """One completed human audit annotation."""

    paper_a: CorpusId
    """Corpus ID of paper A."""
    paper_b: CorpusId
    """Corpus ID of paper B."""
    source: str
    """Pair source label."""
    human_related: bool
    """Human related/unrelated label."""
    model_predictions: dict[str, ModelPrediction]
    """Predictions from all compared models."""


@dataclass(frozen=True, kw_only=True)
class AuditSummary:
    """Aggregate agreement summary for one model."""

    model_name: str
    """Human-readable model name."""
    labelled_pairs: int
    """Number of human-labelled pairs included in the summary."""
    agreement: float
    """Fraction of pairs where the model matches the human label."""


@dataclass(frozen=True, kw_only=True)
class PaperMetadataSummary:
    """Display metadata for one paper in the audit UI."""

    title: str | None
    """Paper title if available."""
    year: int | None
    """Publication year if available."""
    venue: str | None
    """Venue name if available."""


def _infer_paper_metadata_path(facets_path: Path) -> Path | None:
    """Infer the paper-metadata path from the facets path.

    Args:
        facets_path: Path to the facets JSONL file.

    Returns:
        Path to `paper_metadata.json` if it can be inferred and exists, else
        `None`.
    """
    for parent in facets_path.parents:
        candidate = parent / "papers" / "paper_metadata.json"
        if candidate.exists():
            return candidate
    return None


def _load_paper_metadata(
    facets_path: Path,
) -> dict[CorpusId, PaperMetadataSummary]:
    """Load display metadata keyed by corpus ID.

    Args:
        facets_path: Path to the facets JSONL file.

    Returns:
        Mapping from corpus ID to title/year/venue metadata. Returns an empty
        dict if no metadata file can be inferred.
    """
    metadata_path = _infer_paper_metadata_path(facets_path)
    if metadata_path is None:
        return {}

    raw_items = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        CorpusId(int(item["corpusid"])): PaperMetadataSummary(
            title=item.get("title"),
            year=item.get("year"),
            venue=item.get("venue"),
        )
        for item in raw_items
        if item.get("corpusid") is not None
    }


def _panel_title(
    *,
    label: str,
    corpus_id: int,
    metadata: PaperMetadataSummary | None,
) -> str:
    """Build the panel title shown above one paper.

    Args:
        label: Panel label such as `Paper A`.
        corpus_id: Corpus ID for the paper.
        metadata: Optional display metadata.

    Returns:
        Panel title string.
    """
    if metadata is None or not metadata.title:
        return f"{label} ({corpus_id})"
    return f"{label} ({corpus_id}) • {metadata.title}"


def _format_facets_for_audit(facets: Facets) -> str:
    """Format facets for compact side-by-side display.

    Args:
        facets: Facets for one paper.

    Returns:
        Human-readable facet text.
    """
    parts = [f"Corpus ID: {facets.corpusid}"]

    for name, items in (
        ("Problems", facets.problems),
        ("Methods", facets.methods),
        ("Results", facets.results),
        ("Contributions", facets.contributions),
    ):
        if not items:
            continue

        bullet_list = "\n".join(f"- {item}" for item in items)
        parts.append(f"{name}\n{bullet_list}")

    return "\n\n".join(parts)


def _panel_content_width() -> int:
    """Estimate the usable content width for one side-by-side paper panel.

    Returns:
        Approximate character width for text wrapping inside one panel.
    """
    return max((_console.size.width // 2) - 8, 20)


def _window_line_count() -> int:
    """Estimate how many wrapped lines fit in one visible page.

    Returns:
        Number of wrapped lines to show before scrolling is needed.
    """
    return max(_console.size.height - 9, 10)


def _wrap_panel_lines(text: str, *, width: int) -> list[str]:
    """Wrap facet text to the approximate panel width.

    Args:
        text: Unwrapped panel text.
        width: Target wrapping width.

    Returns:
        Wrapped display lines.
    """
    wrapped_lines: list[str] = []

    for line in text.splitlines():
        if not line:
            wrapped_lines.append("")
            continue

        subsequent_indent = "  " if line.startswith("- ") else ""
        chunks = textwrap.wrap(
            line,
            width=width,
            subsequent_indent=subsequent_indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_lines.extend(chunks or [""])

    return wrapped_lines


def _visible_panel_text(
    wrapped_lines: list[str],
    *,
    offset: int,
    window_lines: int,
) -> str:
    """Render the currently visible slice of wrapped panel lines.

    Args:
        wrapped_lines: Fully wrapped lines for one paper.
        offset: Current top-line offset.
        window_lines: Number of wrapped lines visible at once.

    Returns:
        Joined visible panel text.
    """
    visible = wrapped_lines[offset : offset + window_lines]
    return "\n".join(visible) if visible else ""


def _load_named_judgements(files: list[str]) -> dict[str, dict[str, JudgedPair]]:
    """Load named judgement files keyed by `"paper_a|paper_b"`.

    Args:
        files: Input entries in `name=path` format.

    Returns:
        Mapping from model name to keyed judgements.
    """
    named_judgements: dict[str, dict[str, JudgedPair]] = {}

    for entry in files:
        if "=" not in entry:
            die(f"Expected name=path format, got: {entry!r}")

        name, path_str = entry.split("=", 1)
        path = Path(path_str).expanduser()
        if not path.exists():
            die(f"File not found: {path}")

        named_judgements[name] = {
            f"{pair.paper_a}|{pair.paper_b}": pair for pair in load_judged_pairs(path)
        }

    if len(named_judgements) < 2:
        die("Need at least two judgement files for an audit.")

    return named_judgements


def _sample_common_keys(
    named_judgements: dict[str, dict[str, JudgedPair]],
    *,
    count: int,
    seed: int,
) -> list[str]:
    """Sample common pair keys across all input judge files.

    Args:
        named_judgements: Loaded judgement sets by model name.
        count: Number of pairs to sample.
        seed: Sampling seed.

    Returns:
        Sampled pair keys.
    """
    common_keys: set[str] | None = None
    for judgements in named_judgements.values():
        keys = set(judgements)
        common_keys = keys if common_keys is None else (common_keys & keys)

    assert common_keys is not None
    available = sorted(common_keys)
    if not available:
        return []

    rng = random.Random(seed)
    sample_size = min(count, len(available))
    return rng.sample(available, sample_size)


def _load_existing_records(path: Path) -> list[HumanAuditRecord]:
    """Load existing audit records for resume support.

    Args:
        path: Output JSONL path.

    Returns:
        Previously written audit records.
    """
    if not path.exists():
        return []

    return [
        HumanAuditRecord.model_validate_json(line)
        for line_ in path.read_text(encoding="utf-8").splitlines()
        if (line := line_.strip())
    ]


def _record_key(record: HumanAuditRecord) -> str:
    """Build the canonical pair key for one human audit record.

    Args:
        record: Human audit record.

    Returns:
        Pair key in `"paper_a|paper_b"` form.
    """
    return f"{record.paper_a}|{record.paper_b}"


def _save_records(path: Path, records: list[HumanAuditRecord]) -> None:
    """Rewrite the output JSONL file with the latest record per pair.

    Args:
        path: Output JSONL path.
        records: Records to write.
    """
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")


def _display_pair(
    *,
    pair_index: int,
    total_pairs: int,
    pair: JudgedPair,
    metadata_a: PaperMetadataSummary | None,
    metadata_b: PaperMetadataSummary | None,
    wrapped_a: list[str],
    wrapped_b: list[str],
    scroll_offset: int,
    window_lines: int,
    stored_label: bool | None,
    labelled_count: int,
) -> None:
    """Render the two papers side by side for annotation.

    Args:
        pair_index: One-based pair index in the session.
        total_pairs: Total pairs in the current session.
        pair: Reference pair metadata.
        metadata_a: Optional metadata for paper A.
        metadata_b: Optional metadata for paper B.
        wrapped_a: Wrapped display lines for paper A.
        wrapped_b: Wrapped display lines for paper B.
        scroll_offset: Current top-line offset for both panels.
        window_lines: Number of wrapped lines to show in one page.
        stored_label: Existing stored human label for this pair, if any.
        labelled_count: Number of sampled pairs currently labelled.
    """
    _console.rule(f"Pair {pair_index}/{total_pairs} • source={pair.source}")
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_row(
        Panel(
            _visible_panel_text(
                wrapped_a,
                offset=scroll_offset,
                window_lines=window_lines,
            ),
            title=_panel_title(
                label="Paper A",
                corpus_id=pair.paper_a,
                metadata=metadata_a,
            ),
            subtitle=(
                None
                if metadata_a is None
                else f"Year: {metadata_a.year or 'n/a'} • Venue: {metadata_a.venue or 'n/a'}"
            ),
            height=window_lines + 4,
            expand=True,
        ),
        Panel(
            _visible_panel_text(
                wrapped_b,
                offset=scroll_offset,
                window_lines=window_lines,
            ),
            title=_panel_title(
                label="Paper B",
                corpus_id=pair.paper_b,
                metadata=metadata_b,
            ),
            subtitle=(
                None
                if metadata_b is None
                else f"Year: {metadata_b.year or 'n/a'} • Venue: {metadata_b.venue or 'n/a'}"
            ),
            height=window_lines + 4,
            expand=True,
        ),
    )
    _console.print(table)
    if stored_label is None:
        _console.print(
            f"Stored label: [dim]none[/dim] • Labelled {labelled_count}/{total_pairs}"
        )
    else:
        _console.print(
            f"Stored label: {_colour_related_label(stored_label)} • "
            f"Labelled {labelled_count}/{total_pairs}"
        )
    _console.print(
        "Controls: "
        "[cyan]Ctrl-D/Ctrl-U[/cyan] page down/up, "
        "[cyan]g[/cyan] top, "
        "[cyan]G[/cyan] bottom, "
        "[cyan]P/N[/cyan] or [cyan]p/n[/cyan] previous/next pair, "
        "[green]r[/green] related, "
        "[red]u[/red] unrelated, "
        "[yellow]q[/yellow] quit"
    )


def _read_audit_key() -> str:
    """Read one raw keypress for the interactive audit.

    Returns:
        Single key string or control character.
    """
    if not sys.stdin.isatty():
        die("Interactive audit requires a TTY.")

    print("Action: ", end="", flush=True)
    fd = sys.stdin.fileno()
    original_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)


def _confirm_quit() -> bool:
    """Ask for confirmation before quitting the audit loop.

    Returns:
        `True` if the user confirmed quitting.
    """
    choice = _console.input(
        "[yellow]Quit this audit session?[/yellow] [bold](y/N)[/bold]: "
    ).strip()
    return choice.lower() in {"y", "yes"}


def _print_predictions(
    *,
    model_predictions: dict[str, ModelPrediction],
    human_related: bool,
) -> None:
    """Print model predictions after a human label is collected.

    Args:
        model_predictions: Model predictions for the current pair.
        human_related: Human label for the current pair.
    """
    _console.print(f"\n[bold]Your label:[/bold] {_colour_related_label(human_related)}")

    table = Table(title="Model Predictions", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Related")
    table.add_column("Confidence", justify="right")
    table.add_column("Type")
    table.add_column("Reasoning")

    for name, prediction in model_predictions.items():
        table.add_row(
            name,
            _colour_related_label(prediction.related),
            f"{prediction.confidence:.2f}",
            prediction.relationship_type,
            prediction.reasoning,
        )

    _console.print(table)


def _all_models_agree_against_human(record: HumanAuditRecord) -> bool:
    """Return whether all model labels agree and all are wrong.

    Args:
        record: Human audit record.

    Returns:
        `True` if every model predicts the same label and that label differs
        from the human label.
    """
    model_labels = {
        prediction.related for prediction in record.model_predictions.values()
    }
    return len(model_labels) == 1 and next(iter(model_labels)) != record.human_related


def _build_audit_summaries(
    records: list[HumanAuditRecord],
) -> tuple[list[AuditSummary], int]:
    """Build per-model agreement summaries for completed audit records.

    Args:
        records: Completed human audit records.

    Returns:
        Tuple of (per-model summaries, unanimous-model-miss count).
    """
    if not records:
        return [], 0

    model_names = sorted(records[0].model_predictions)
    summaries: list[AuditSummary] = []

    for model_name in model_names:
        matches = sum(
            record.model_predictions[model_name].related == record.human_related
            for record in records
        )
        summaries.append(
            AuditSummary(
                model_name=model_name,
                labelled_pairs=len(records),
                agreement=matches / len(records),
            )
        )

    unanimous_misses = sum(
        _all_models_agree_against_human(record) for record in records
    )
    return summaries, unanimous_misses


def run_interactive_human_audit(
    *,
    facets_path: Path,
    output_path: Path,
    files: list[str],
    count: int,
    seed: int,
) -> None:
    """Run an interactive human audit session over common judged pairs.

    Args:
        facets_path: Path to facets JSONL file.
        output_path: Output JSONL path for saved human audit records.
        files: Judgement files in `name=path` format.
        count: Maximum number of new pairs to audit this session.
        seed: Sampling seed.
    """
    named_judgements = _load_named_judgements(files)
    existing_records = _load_existing_records(output_path)
    records_by_key = {_record_key(record): record for record in existing_records}
    sampled_keys = _sample_common_keys(
        named_judgements,
        count=count,
        seed=seed,
    )

    if not sampled_keys:
        _console.print("[yellow]No new common pairs available for audit.[/yellow]")
        summaries, unanimous_misses = _build_audit_summaries(existing_records)
        _print_audit_summary(summaries, unanimous_misses)
        return

    facets_by_id = load_facets_map(facets_path)
    metadata_by_id = _load_paper_metadata(facets_path)
    _console.print(
        f"Loaded {len(named_judgements)} judge files and sampled"
        f" {len(sampled_keys)} pairs."
    )

    session_keys = list(sampled_keys)
    current_index = 0

    while session_keys:
        key = session_keys[current_index]
        reference_pair = next(iter(named_judgements.values()))[key]
        facets_a = facets_by_id.get(reference_pair.paper_a)
        facets_b = facets_by_id.get(reference_pair.paper_b)
        if facets_a is None or facets_b is None:
            _console.print(
                f"[yellow]Skipping {key}: missing facets for one or both papers.[/yellow]"
            )
            session_keys.pop(current_index)
            if session_keys:
                current_index = min(current_index, len(session_keys) - 1)
            continue

        wrapped_a = _wrap_panel_lines(
            _format_facets_for_audit(facets_a),
            width=_panel_content_width(),
        )
        wrapped_b = _wrap_panel_lines(
            _format_facets_for_audit(facets_b),
            width=_panel_content_width(),
        )
        window_lines = _window_line_count()
        max_offset = max(0, max(len(wrapped_a), len(wrapped_b)) - window_lines)
        scroll_offset = 0

        human_related: bool | None = None
        navigated = False
        action = ""
        while True:
            _console.clear()
            labelled_count = sum(key_ in records_by_key for key_ in session_keys)
            _display_pair(
                pair_index=current_index + 1,
                total_pairs=len(session_keys),
                pair=reference_pair,
                metadata_a=metadata_by_id.get(reference_pair.paper_a),
                metadata_b=metadata_by_id.get(reference_pair.paper_b),
                wrapped_a=wrapped_a,
                wrapped_b=wrapped_b,
                scroll_offset=scroll_offset,
                window_lines=window_lines,
                stored_label=(
                    None
                    if key not in records_by_key
                    else records_by_key[key].human_related
                ),
                labelled_count=labelled_count,
            )
            _console.print(
                f"Showing lines {scroll_offset + 1}-"
                f"{min(scroll_offset + window_lines, max(len(wrapped_a), len(wrapped_b)))} "
                f"of {max(len(wrapped_a), len(wrapped_b))}"
            )

            action = _read_audit_key()
            lowered = action.lower()
            if lowered in {"r", "related"}:
                human_related = True
                break
            if lowered in {"u", "unrelated"}:
                human_related = False
                break
            if lowered in {"q", "quit"}:
                if _confirm_quit():
                    human_related = None
                    break
                continue
            if action == "\x04":
                scroll_offset = min(scroll_offset + window_lines, max_offset)
                continue
            if action == "\x15":
                scroll_offset = max(scroll_offset - window_lines, 0)
                continue
            if action == "g":
                scroll_offset = 0
                continue
            if action == "G":
                scroll_offset = max_offset
                continue
            if action in {"P", "p"}:
                current_index = max(current_index - 1, 0)
                navigated = True
                break
            if action in {"N", "n"}:
                current_index = min(current_index + 1, len(session_keys) - 1)
                navigated = True
                break

            _console.print(
                "[yellow]Use Ctrl-D, Ctrl-U, g, G, P, N, r, u, or q.[/yellow]"
            )

        if navigated:
            continue
        if human_related is None:
            break

        model_predictions = {
            name: ModelPrediction(
                related=judgement.related,
                confidence=judgement.confidence,
                relationship_type=judgement.relationship_type.value,
                reasoning=judgement.reasoning,
            )
            for name, judgements in named_judgements.items()
            if (judgement := judgements[key])
        }
        record = HumanAuditRecord(
            paper_a=reference_pair.paper_a,
            paper_b=reference_pair.paper_b,
            source=reference_pair.source,
            human_related=human_related,
            model_predictions=model_predictions,
        )
        records_by_key[key] = record
        _save_records(output_path, list(records_by_key.values()))

        _print_predictions(
            model_predictions=model_predictions,
            human_related=human_related,
        )
        if _all_models_agree_against_human(record):
            _console.print(
                "[bold red]Warning:[/bold red] all compared models agree with each "
                "other, and all disagree with your label."
            )
        current_index = min(current_index + 1, len(session_keys) - 1)

    summaries, unanimous_misses = _build_audit_summaries(list(records_by_key.values()))
    _print_audit_summary(summaries, unanimous_misses)


def _print_audit_summary(
    summaries: list[AuditSummary],
    unanimous_misses: int,
) -> None:
    """Print the final audit agreement summary.

    Args:
        summaries: Per-model summaries.
        unanimous_misses: Count of unanimous model misses against the human.
    """
    if not summaries:
        _console.print("[yellow]No completed annotations yet.[/yellow]")
        return

    table = Table(title="Human Audit Summary", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Labelled Pairs", justify="right")
    table.add_column("Agreement", justify="right")

    for summary in summaries:
        table.add_row(
            summary.model_name,
            str(summary.labelled_pairs),
            f"{summary.agreement:.1%}",
        )

    _console.print(table)
    _console.print(
        f"Unanimous model disagreements against human: [bold]{unanimous_misses}[/bold]"
    )
