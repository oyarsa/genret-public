"""Compute groundedness and inter-annotator agreement over returned sheets.

Takes the completed facet-annotation CSVs (one per annotator, as produced by
`scripts/eval/sample_facet_annotation.py` and filled in by hand) and reports:

- per-annotator score distributions, means, and share at or above the
  headline threshold, broken down by facet type;
- the pooled headline metric, averaging multiply-rated items across their
  annotators before thresholding;
- agreement over the shared core: ordinal Krippendorff's alpha across all
  annotators, mean pairwise linearly weighted kappa, and per-pair raw,
  within-one, and threshold agreement, each with a percentile bootstrap
  interval;
- the structure of the disagreements: signed direction per pair (with a sign
  test for systematic leniency) and clustering by paper.

Sheets need only share an `item_hash` column plus `validity`; extra columns
(`confidence`, `rationale_detail`, ...) are ignored, so sheets returned in
different formats can be compared directly. Any partial-overlap design works:
agreement is computed over whichever items happen to carry two or more
ratings, so this handles the full-overlap and disjoint-split cases too.

Chance-corrected coefficients are reported as `degenerate` rather than as a
number whenever an annotator's ratings are constant over the compared items.
Kappa is then exactly zero and alpha undefined, as artefacts of the margins
rather than as measurements -- a real risk on this scale, where most items
sit at 4-5.

Usage:
    uv run python scripts/eval/facet_annotation_agreement.py \
        output/genret/facet_annotation/done/*.csv

Sheets are labelled A, B, C, ... in argument order unless `--label` is given
once per sheet, in the same order.
"""

from __future__ import annotations

import csv
import itertools
import math
import random
import re
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

BOOTSTRAP_RESAMPLES = 5000
"""Percentile bootstrap resamples used for every interval."""
MIN_BOOTSTRAP_YIELD = 0.5
"""Fraction of resamples that must yield a defined statistic for a usable CI."""
ITEM_HASH_WIDTH = 12
"""Width of `item_hash`, a truncated SHA-256 hex digest from the sampler."""
SCIENTIFIC_NOTATION = re.compile(r"^\d(?:\.\d+)?[eE][+-]?\d+$")
"""Shape of a hash a spreadsheet has irrecoverably coerced to a float."""


def normalise_item_hash(value: str, path: Path, lineno: int) -> tuple[str, bool]:
    """Repair an `item_hash` a spreadsheet has coerced to a number.

    An all-digit hash round-tripped through Excel or Sheets loses its leading
    zeros, which silently breaks the join between sheets. Padding is safe:
    the hash is fixed-width hexadecimal, so the original is recoverable.

    Args:
        value: Raw `item_hash` cell.
        path: Sheet the value came from, for error messages.
        lineno: Line the value came from, for error messages.

    Returns:
        The hash and whether it had to be repaired.

    Raises:
        ValueError: If the hash is too long, or was coerced to scientific
            notation, in which case the original digits are unrecoverable.
    """
    if len(value) == ITEM_HASH_WIDTH:
        return value, False
    if SCIENTIFIC_NOTATION.match(value):
        raise ValueError(
            f"{path}:{lineno}: item_hash {value!r} was coerced to scientific "
            "notation by a spreadsheet and cannot be recovered; re-export the "
            "sheet with the column formatted as text."
        )
    if len(value) < ITEM_HASH_WIDTH and value.isdigit():
        return value.zfill(ITEM_HASH_WIDTH), True
    raise ValueError(
        f"{path}:{lineno}: item_hash {value!r} is not a "
        f"{ITEM_HASH_WIDTH}-character hex digest."
    )


@dataclass(frozen=True, kw_only=True)
class Item:
    """A single rated facet item, as identified across every sheet."""

    item_hash: str
    """Stable hash of (paper title, facet type, facet item)."""
    paper_title: str
    """Title of the paper the item was extracted from."""
    facet_type: str
    """Facet type of the item (`problems`, `methods`, ...)."""


@dataclass(frozen=True, kw_only=True)
class Sheet:
    """One annotator's returned sheet."""

    label: str
    """Short display name for the annotator."""
    path: Path
    """Source CSV the sheet was read from."""
    ratings: Mapping[str, int]
    """Score per `item_hash`."""


@dataclass(frozen=True, kw_only=True)
class PairAgreement:
    """Agreement between one pair of annotators over their shared items."""

    left: str
    """Label of the first annotator."""
    right: str
    """Label of the second annotator."""
    pairs: Sequence[tuple[int, int]]
    """Aligned `(left score, right score)` for every shared item."""
    kappa: float | None
    """Linearly weighted kappa, or `None` when the margins are degenerate."""
    kappa_ci: tuple[float, float] | None
    """Bootstrap interval for `kappa`, when estimable."""
    binary_kappa: float | None
    """Kappa on the thresholded decision, or `None` when degenerate."""
    binary_kappa_ci: tuple[float, float] | None
    """Bootstrap interval for `binary_kappa`, when estimable."""


def load_sheet(path: Path, label: str) -> tuple[Sheet, dict[str, Item]]:
    """Read one annotator's CSV.

    Args:
        path: Completed annotation CSV.
        label: Display name for this annotator.

    Returns:
        The sheet, and the item metadata it carries, keyed by `item_hash`.

    Raises:
        ValueError: If a row is missing `item_hash`, or its `validity` is
            blank or not an integer.
    """
    ratings: dict[str, int] = {}
    items: dict[str, Item] = {}
    repaired = 0
    with path.open(newline="", encoding="utf-8") as f:
        for lineno, row in enumerate(csv.DictReader(f), start=2):
            raw_hash = (row.get("item_hash") or "").strip()
            if not raw_hash:
                raise ValueError(f"{path}:{lineno}: missing item_hash.")
            item_hash, was_repaired = normalise_item_hash(raw_hash, path, lineno)
            repaired += was_repaired
            raw = (row.get("validity") or "").strip()
            if not raw:
                raise ValueError(f"{path}:{lineno}: blank validity for {item_hash}.")
            try:
                ratings[item_hash] = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{lineno}: non-integer validity {raw!r} for {item_hash}."
                ) from exc
            items[item_hash] = Item(
                item_hash=item_hash,
                paper_title=(row.get("paper_title") or "").strip(),
                facet_type=(row.get("facet_type") or "").strip(),
            )
    if repaired:
        print(
            f"Warning: {path.name}: restored leading zeros on {repaired} "
            "item_hash value(s) stripped by a spreadsheet."
        )
    return Sheet(label=label, path=path, ratings=ratings), items


def weighted_kappa(
    pairs: Sequence[tuple[int, int]], categories: Sequence[int]
) -> float | None:
    """Cohen's linearly weighted kappa.

    Weights are `1 - |i - j| / (max - min)`, so adjacent-category
    disagreements on an ordinal scale keep most of their credit.

    Args:
        pairs: Aligned rating pairs from two annotators.
        categories: Full ordered scale, including unused levels.

    Returns:
        The coefficient, or `None` when it is degenerate: either annotator
        rating every item identically forces kappa to zero regardless of the
        data, which is an artefact rather than a measurement.
    """
    if not pairs:
        return None
    if len({left for left, _ in pairs}) == 1 or len({right for _, right in pairs}) == 1:
        return None
    span = categories[-1] - categories[0]
    if span == 0:
        return None

    index = {category: i for i, category in enumerate(categories)}
    size = len(categories)
    joint = [[0.0] * size for _ in range(size)]
    for left, right in pairs:
        joint[index[left]][index[right]] += 1 / len(pairs)
    marginal_left = [sum(row) for row in joint]
    marginal_right = [sum(row[j] for row in joint) for j in range(size)]

    observed = expected = 0.0
    for i, j in itertools.product(range(size), repeat=2):
        weight = 1 - abs(categories[i] - categories[j]) / span
        observed += weight * joint[i][j]
        expected += weight * marginal_left[i] * marginal_right[j]
    if abs(1 - expected) < 1e-12:
        return None
    return (observed - expected) / (1 - expected)


def krippendorff_alpha(
    units: Sequence[Sequence[int]], categories: Sequence[int]
) -> float | None:
    """Krippendorff's alpha with the ordinal difference function.

    Unlike weighted kappa, this takes any number of annotators per unit,
    tolerates missing ratings, and derives category distances from the
    observed marginal distribution rather than assuming equal spacing.

    Args:
        units: One sequence of ratings per item; units with fewer than two
            ratings are ignored, as they carry no agreement information.
        categories: Full ordered scale, including unused levels.

    Returns:
        The coefficient, or `None` when expected disagreement is zero (every
        pairable rating identical), which leaves alpha undefined.
    """
    pairable = [unit for unit in units if len(unit) >= 2]
    counts = Counter(rating for unit in pairable for rating in unit)
    total = sum(counts.values())
    if total < 2:
        return None

    position = {category: i for i, category in enumerate(categories)}

    def delta_squared(left: int, right: int) -> float:
        """Ordinal distance between two categories, given the marginals."""
        low, high = sorted((position[left], position[right]))
        spanned = sum(counts[categories[g]] for g in range(low, high + 1))
        return (spanned - (counts[left] + counts[right]) / 2) ** 2

    observed = 0.0
    for unit in pairable:
        ordered_pairs = (
            sum(
                delta_squared(left, right)
                for left, right in itertools.combinations(unit, 2)
            )
            * 2
        )
        observed += ordered_pairs / (len(unit) - 1)
    observed /= total

    expected = sum(
        counts[left] * counts[right] * delta_squared(left, right)
        for left, right in itertools.permutations(categories, 2)
    ) / (total * (total - 1))
    if expected == 0:
        return None
    return 1 - observed / expected


def gwet_ac1(units: Sequence[Sequence[int]], categories: Sequence[int]) -> float | None:
    """Gwet's AC1, a chance-corrected coefficient robust to skewed prevalence.

    Kappa and alpha estimate chance agreement from the observed marginals, so
    when nearly every item falls in one category they treat almost all
    observed agreement as accidental and collapse toward zero even though the
    raters plainly concur. AC1 instead estimates chance agreement as the
    probability of a rater being uncertain, which shrinks as prevalence
    becomes more extreme, and so stays interpretable in exactly that case.

    Args:
        units: One sequence of ratings per item; units with fewer than two
            ratings are ignored.
        categories: Full set of categories.

    Returns:
        The coefficient, or `None` when there is nothing to estimate from.
    """
    pairable = [unit for unit in units if len(unit) >= 2]
    if not pairable or len(categories) < 2:
        return None

    observed = 0.0
    prevalence = dict.fromkeys(categories, 0.0)
    for unit in pairable:
        raters = len(unit)
        counts = Counter(unit)
        observed += sum(
            counts[category] * (counts[category] - 1) for category in categories
        ) / (raters * (raters - 1))
        for category in categories:
            prevalence[category] += counts[category] / raters
    observed /= len(pairable)

    shares = {category: total / len(pairable) for category, total in prevalence.items()}
    expected = sum(share * (1 - share) for share in shares.values()) / (
        len(categories) - 1
    )
    if abs(1 - expected) < 1e-12:
        return None
    return (observed - expected) / (1 - expected)


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation here because the proportion sits
    near 1, where the approximation runs past the boundary.

    Args:
        successes: Items meeting the criterion.
        trials: Items rated.

    Returns:
        The 95% interval.
    """
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)


def bootstrap_ci[T](
    sample: Sequence[T],
    statistic: Callable[[Sequence[T]], float | None],
    rng: random.Random,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval, resampling items with replacement.

    Args:
        sample: Items to resample (rating pairs, or per-unit rating lists).
        statistic: Statistic to recompute per resample; may return `None`
            when a resample is degenerate.
        rng: Seeded random source.
        resamples: Number of resamples to draw.

    Returns:
        The 95% interval, or `None` when too many resamples were degenerate
        for the interval to mean anything.
    """
    values: list[float] = []
    for _ in range(resamples):
        drawn = [sample[rng.randrange(len(sample))] for _ in range(len(sample))]
        if (value := statistic(drawn)) is not None:
            values.append(value)
    if len(values) < resamples * MIN_BOOTSTRAP_YIELD:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values))]


def sign_test(successes: int, trials: int) -> float:
    """Two-sided sign test p-value under a fair-coin null.

    Args:
        successes: Disagreements in the more common direction.
        trials: Total disagreements.

    Returns:
        The p-value; `1.0` when there are no disagreements to test.
    """
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    cumulative = sum(math.comb(trials, k) for k in range(tail + 1))
    return min(2 * cumulative / 2**trials, 1.0)


def format_coefficient(
    value: float | None, interval: tuple[float, float] | None
) -> str:
    """Render a coefficient and its interval for the report.

    Args:
        value: The coefficient, or `None` when degenerate.
        interval: Its bootstrap interval, when estimable.

    Returns:
        A display string, flagging degeneracy rather than printing a zero
        that only reflects the margins.
    """
    if value is None:
        return "degenerate (an annotator's ratings are constant here)"
    if interval is None:
        return f"{value:.3f} (interval unstable)"
    return f"{value:.3f}  95% CI [{interval[0]:.3f}, {interval[1]:.3f}]"


def report_marginals(
    sheets: Sequence[Sheet], items: Mapping[str, Item], threshold: int
) -> None:
    """Print per-annotator distributions and per-facet-type breakdowns.

    Args:
        sheets: Loaded annotator sheets.
        items: Item metadata keyed by `item_hash`.
        threshold: Score at or above which an item counts as grounded.
    """
    print("\n== Per annotator ==")
    for sheet in sheets:
        scores = list(sheet.ratings.values())
        at_threshold = sum(1 for score in scores if score >= threshold)
        distribution = " ".join(
            f"{score}:{count}" for score, count in sorted(Counter(scores).items())
        )
        print(f"\n{sheet.label} ({sheet.path.name}) n={len(scores)}")
        print(f"  distribution     : {distribution}")
        print(
            f"  mean / median    : {statistics.mean(scores):.2f} / "
            f"{statistics.median(scores):.1f}"
        )
        low, high = wilson_interval(at_threshold, len(scores))
        print(
            f"  share >= {threshold}       : {at_threshold}/{len(scores)} = "
            f"{at_threshold / len(scores):.1%}  95% CI [{low:.1%}, {high:.1%}]"
        )

        by_type: dict[str, list[int]] = {}
        for item_hash, score in sheet.ratings.items():
            by_type.setdefault(items[item_hash].facet_type, []).append(score)
        for facet_type, values in sorted(by_type.items()):
            hits = sum(1 for value in values if value >= threshold)
            print(
                f"    {facet_type:14s} n={len(values):3d} "
                f"mean={statistics.mean(values):.2f}  >= {threshold}: "
                f"{hits}/{len(values)}"
            )


def report_headline(
    sheets: Sequence[Sheet],
    items: Mapping[str, Item],
    threshold: int,
    rng: random.Random,
) -> None:
    """Print the pooled headline metric over all distinct items.

    Multiply-rated items are averaged across their annotators before
    thresholding, so a shared-core item counts once rather than once per
    annotator.

    Args:
        sheets: Loaded annotator sheets.
        items: Item metadata keyed by `item_hash`.
        threshold: Score at or above which an item counts as grounded.
        rng: Seeded random source for the bootstrap.
    """
    per_item: dict[str, list[int]] = {}
    for sheet in sheets:
        for item_hash, score in sheet.ratings.items():
            per_item.setdefault(item_hash, []).append(score)
    means = {
        item_hash: statistics.mean(scores) for item_hash, scores in per_item.items()
    }
    grounded = sum(1 for mean in means.values() if mean >= threshold)
    ratings_total = sum(len(scores) for scores in per_item.values())
    coverage = Counter(len(scores) for scores in per_item.values())

    print("\n== Pooled ==")
    print(f"  distinct items   : {len(per_item)} ({ratings_total} ratings)")
    print(
        "  raters per item  : "
        + ", ".join(f"{raters}x: {count}" for raters, count in sorted(coverage.items()))
    )
    # Items from one paper share an extraction and a rater's reading of it, so
    # resample whole papers rather than items to keep the interval honest.
    by_paper: dict[str, list[float]] = {}
    for item_hash, mean in means.items():
        by_paper.setdefault(items[item_hash].paper_title, []).append(mean)
    papers = list(by_paper.values())
    share_ci = bootstrap_ci(
        papers,
        lambda drawn: statistics.mean(
            value >= threshold for paper in drawn for value in paper
        ),
        rng,
    )
    interval = ""
    if share_ci is not None:
        interval = f"  95% CI [{share_ci[0]:.1%}, {share_ci[1]:.1%}]"
    print(f"  mean score       : {statistics.mean(means.values()):.2f}")
    print(
        f"  share >= {threshold}       : {grounded}/{len(means)} = "
        f"{grounded / len(means):.1%}{interval}"
    )
    print(
        "                     (multiply-rated items averaged first; "
        "interval clusters by paper)"
    )
    by_type: dict[str, list[float]] = {}
    for item_hash, mean in means.items():
        by_type.setdefault(items[item_hash].facet_type, []).append(mean)
    for facet_type, values in sorted(by_type.items()):
        hits = sum(1 for value in values if value >= threshold)
        print(
            f"    {facet_type:14s} n={len(values):3d} "
            f"mean={statistics.mean(values):.2f}  >= {threshold}: "
            f"{hits}/{len(values)}"
        )


def binarise(pairs: Sequence[tuple[int, int]], threshold: int) -> list[tuple[int, int]]:
    """Reduce rating pairs to the grounded/not-grounded decision.

    Args:
        pairs: Aligned rating pairs from two annotators.
        threshold: Score at or above which an item counts as grounded.

    Returns:
        The same pairs as `1`/`0` indicators.
    """
    return [(int(left >= threshold), int(right >= threshold)) for left, right in pairs]


def compute_pair(
    left: Sheet,
    right: Sheet,
    categories: Sequence[int],
    threshold: int,
    rng: random.Random,
) -> PairAgreement | None:
    """Compute one pair's agreement over the items both annotators rated.

    Reports the coefficient on the full scale and on the thresholded
    decision: the two answer different questions, and the headline metric
    rests on the latter.

    Args:
        left: First annotator's sheet.
        right: Second annotator's sheet.
        categories: Full ordered scale.
        threshold: Score at or above which an item counts as grounded.
        rng: Seeded random source for the bootstrap.

    Returns:
        The pair's agreement, or `None` when the two share no items.
    """
    shared = sorted(set(left.ratings) & set(right.ratings))
    if not shared:
        return None
    pairs = [(left.ratings[h], right.ratings[h]) for h in shared]
    kappa = weighted_kappa(pairs, categories)
    interval = None
    if kappa is not None:
        interval = bootstrap_ci(
            pairs, lambda drawn: weighted_kappa(drawn, categories), rng
        )

    binary_pairs = binarise(pairs, threshold)
    binary_kappa = weighted_kappa(binary_pairs, (0, 1))
    binary_interval = None
    if binary_kappa is not None:
        binary_interval = bootstrap_ci(
            binary_pairs, lambda drawn: weighted_kappa(drawn, (0, 1)), rng
        )
    return PairAgreement(
        left=left.label,
        right=right.label,
        pairs=pairs,
        kappa=kappa,
        kappa_ci=interval,
        binary_kappa=binary_kappa,
        binary_kappa_ci=binary_interval,
    )


def report_pair(agreement: PairAgreement, threshold: int) -> None:
    """Print one pair's raw agreement, coefficient, and disagreement direction.

    Args:
        agreement: The pair's computed agreement.
        threshold: Score at or above which an item counts as grounded.
    """
    pairs = agreement.pairs
    exact = sum(1 for left, right in pairs if left == right)
    within_one = sum(1 for left, right in pairs if abs(left - right) <= 1)
    on_threshold = sum(
        1 for left, right in pairs if (left >= threshold) == (right >= threshold)
    )
    disagreements = [(left, right) for left, right in pairs if left != right]
    left_higher = sum(1 for left, right in disagreements if left > right)

    print(f"\n{agreement.left} vs {agreement.right}  (n={len(pairs)})")
    print(f"  exact              : {exact}/{len(pairs)} = {exact / len(pairs):.1%}")
    print(
        f"  within one point   : {within_one}/{len(pairs)} = "
        f"{within_one / len(pairs):.1%}"
    )
    print(
        f"  on the >= {threshold} split   : {on_threshold}/{len(pairs)} = "
        f"{on_threshold / len(pairs):.1%}"
    )
    print(
        f"  mean |difference|  : "
        f"{statistics.mean(abs(left - right) for left, right in pairs):.2f}"
    )
    print(
        f"  mean {agreement.left} / mean {agreement.right}: "
        f"{statistics.mean(left for left, _ in pairs):.2f} / "
        f"{statistics.mean(right for _, right in pairs):.2f}"
    )
    print(
        f"  weighted kappa     : "
        f"{format_coefficient(agreement.kappa, agreement.kappa_ci)}"
    )
    print(
        f"  kappa on >= {threshold}       : "
        f"{format_coefficient(agreement.binary_kappa, agreement.binary_kappa_ci)}"
    )
    if disagreements:
        p_value = sign_test(left_higher, len(disagreements))
        print(
            f"  direction          : {agreement.left} higher in "
            f"{left_higher}/{len(disagreements)} disagreements "
            f"(sign test p={p_value:.4f})"
        )


def report_disagreement_clustering(
    sheets: Sequence[Sheet], items: Mapping[str, Item]
) -> None:
    """Print how disagreements distribute over papers.

    Disagreement concentrated in a few papers points at differing readings of
    those papers rather than at inconsistent use of the scale.

    Args:
        sheets: Loaded annotator sheets.
        items: Item metadata keyed by `item_hash`.
    """
    per_item: dict[str, list[int]] = {}
    for sheet in sheets:
        for item_hash, score in sheet.ratings.items():
            per_item.setdefault(item_hash, []).append(score)
    multiply_rated = {h: s for h, s in per_item.items() if len(s) >= 2}
    if not multiply_rated:
        return

    rated = Counter(items[h].paper_title for h in multiply_rated)
    split = Counter(
        items[h].paper_title for h, s in multiply_rated.items() if len(set(s)) > 1
    )
    total_split = sum(split.values())
    print(
        f"\n== Disagreement clustering ==\n  {total_split}/{len(multiply_rated)} "
        f"multiply-rated items are not unanimous, over "
        f"{len(split)}/{len(rated)} papers"
    )
    for title, count in split.most_common():
        print(f"    {count}/{rated[title]}  {title[:64]}")


def report_agreement(
    sheets: Sequence[Sheet],
    items: Mapping[str, Item],
    categories: Sequence[int],
    threshold: int,
    rng: random.Random,
) -> None:
    """Print alpha across all annotators plus every pairwise breakdown.

    Args:
        sheets: Loaded annotator sheets.
        items: Item metadata keyed by `item_hash`.
        categories: Full ordered scale.
        threshold: Score at or above which an item counts as grounded.
        rng: Seeded random source for the bootstrap.
    """
    per_item: dict[str, list[int]] = {}
    for sheet in sheets:
        for item_hash, score in sheet.ratings.items():
            per_item.setdefault(item_hash, []).append(score)
    units = [scores for scores in per_item.values() if len(scores) >= 2]
    if not units:
        print("\n== Agreement ==\n  No item was rated by more than one annotator.")
        return

    print(
        f"\n== Agreement ==\n  shared items: {len(units)} "
        f"({sum(len(u) for u in units)} ratings)"
    )
    alpha = krippendorff_alpha(units, categories)
    interval = None
    if alpha is not None:
        interval = bootstrap_ci(
            units, lambda drawn: krippendorff_alpha(drawn, categories), rng
        )
    print(f"  ordinal alpha    : {format_coefficient(alpha, interval)}")

    # With two categories every disagreement carries the same distance, so the
    # ordinal difference function reduces to the nominal one.
    binary_units = [[int(score >= threshold) for score in unit] for unit in units]
    binary_alpha = krippendorff_alpha(binary_units, (0, 1))
    binary_interval = None
    if binary_alpha is not None:
        binary_interval = bootstrap_ci(
            binary_units, lambda drawn: krippendorff_alpha(drawn, (0, 1)), rng
        )
    print(
        f"  alpha on >= {threshold}     : "
        f"{format_coefficient(binary_alpha, binary_interval)}"
    )
    ac1 = gwet_ac1(binary_units, (0, 1))
    ac1_interval = None
    if ac1 is not None:
        ac1_interval = bootstrap_ci(
            binary_units, lambda drawn: gwet_ac1(drawn, (0, 1)), rng
        )
    print(f"  AC1 on >= {threshold}       : {format_coefficient(ac1, ac1_interval)}")
    unanimous = sum(1 for unit in units if len(set(unit)) == 1)
    print(
        f"  unanimous items  : {unanimous}/{len(units)} = {unanimous / len(units):.1%}"
    )
    binary_unanimous = sum(1 for unit in binary_units if len(set(unit)) == 1)
    print(
        f"  unanimous on >= {threshold} : {binary_unanimous}/{len(units)} = "
        f"{binary_unanimous / len(units):.1%}"
    )

    agreements = [
        agreement
        for left, right in itertools.combinations(sheets, 2)
        if (agreement := compute_pair(left, right, categories, threshold, rng))
        is not None
    ]
    if defined := [a.kappa for a in agreements if a.kappa is not None]:
        print(
            f"  mean pairwise kappa: {statistics.mean(defined):.3f} "
            f"(over {len(defined)}/{len(agreements)} estimable pairs)"
        )
    for agreement in agreements:
        report_pair(agreement, threshold)
    report_disagreement_clustering(sheets, items)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode=None,
    pretty_exceptions_show_locals=False,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    sheet_paths: Annotated[
        list[Path],
        typer.Argument(
            show_default=False,
            metavar="SHEETS...",
            help="One completed annotation CSV per annotator.",
        ),
    ],
    label: Annotated[
        list[str] | None,
        typer.Option(
            "--label",
            help="Display name per annotator, in argument order. "
            "Defaults to A, B, C, ...",
        ),
    ] = None,
    threshold: Annotated[
        int, typer.Option(help="Score at or above which an item counts as grounded.")
    ] = 4,
    min_score: Annotated[int, typer.Option(help="Lowest point on the scale.")] = 1,
    max_score: Annotated[int, typer.Option(help="Highest point on the scale.")] = 5,
    seed: Annotated[int, typer.Option(help="Random seed for the bootstrap.")] = 0,
) -> None:
    """Report groundedness and agreement across the given annotation sheets."""
    labels = label or [chr(ord("A") + i) for i in range(len(sheet_paths))]
    if len(labels) != len(sheet_paths):
        raise ValueError(
            f"Got {len(labels)} --label values for {len(sheet_paths)} sheets."
        )

    sheets: list[Sheet] = []
    items: dict[str, Item] = {}
    for path, name in zip(sheet_paths, labels, strict=True):
        sheet, sheet_items = load_sheet(path, name)
        sheets.append(sheet)
        items.update(sheet_items)

    categories = list(range(min_score, max_score + 1))
    if off_scale := {
        score
        for sheet in sheets
        for score in sheet.ratings.values()
        if score not in categories
    }:
        raise ValueError(
            f"Scores {sorted(off_scale)} fall outside the "
            f"{min_score}-{max_score} scale; adjust --min-score/--max-score."
        )

    rng = random.Random(seed)
    report_marginals(sheets, items, threshold)
    report_headline(sheets, items, threshold, rng)
    report_agreement(sheets, items, categories, threshold, rng)


if __name__ == "__main__":
    app()
