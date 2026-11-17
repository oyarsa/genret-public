"""Investigate S2 diffs API to evaluate 2025 NLP paper coverage.

Steps 1-3 of the "Add 2025 Papers" plan:
1. Enumerate releases between 2026-02-03 and 2026-03-10
2. Fetch diff metadata for the `papers` dataset
3. Download diffs, filter for NLP venue + 2025 year, count new papers

Usage:
    uv run python scripts/investigate_s2_diffs.py [--download]

Without --download: just queries API for release list and diff file counts.
With --download: downloads and processes diff files to count 2025 NLP papers.
"""

import argparse
import gzip
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx

from genret.types import is_type

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATASETS_API = "https://api.semanticscholar.org/datasets/v1"
BASE_RELEASE = "2026-02-03"
TARGET_RELEASE = "2026-03-10"

# Same venue patterns as the pipeline
VENUE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bACL\b",
        r"\bEMNLP\b",
        r"\bNAACL\b",
        r"\bCOLING\b",
        r"\bEACL\b",
        r"\bAACL\b",
        r"\bCONLL\b",
        r"\bSemEval\b",
        r"\b\*SEM\b",
        r"\bTACL\b",
        r"\bCL\b",
    ]
]


def matches_nlp_venue(record: dict[str, Any]) -> bool:
    """Check if a paper record matches NLP venue patterns or has ACL ID."""
    external_ids: dict[str, str] = record.get("externalids") or {}
    assert is_type(external_ids, dict[str, str])
    if external_ids.get("ACL") is not None:
        return True
    venue: str = record.get("venue") or ""
    return any(p.search(venue) for p in VENUE_PATTERNS)


def is_2025_paper(record: dict[str, Any]) -> bool:
    """Check if paper is from 2025."""
    return record.get("year") == 2025


def get_headers() -> dict[str, str]:
    """Return S2 API auth headers if an API key is set."""
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if api_key:
        return {"x-api-key": api_key}
    return {}


def list_releases(client: httpx.Client) -> list[str]:
    """List all available releases."""
    resp = client.get(f"{DATASETS_API}/release")
    resp.raise_for_status()
    return resp.json()


def get_diffs(
    client: httpx.Client, from_release: str, to_release: str, dataset: str
) -> list[dict[str, Any]] | dict[str, Any]:
    """Get diff info between two releases for a dataset."""
    url = f"{DATASETS_API}/diffs/{from_release}/to/{to_release}/{dataset}"
    resp = client.get(url, headers=get_headers())
    resp.raise_for_status()
    return resp.json()


def download_and_count_diff_file(
    client: httpx.Client,
    url: str,
    existing_nlp_ids: set[int],
) -> tuple[int, int, int, set[int]]:
    """Download a diff file and count 2025 NLP papers.

    Returns (total_records, nlp_2025_new, nlp_2025_updated, new_nlp_ids).
    """
    total = 0
    nlp_2025_new = 0
    nlp_2025_updated = 0
    new_ids: set[int] = set()

    with tempfile.NamedTemporaryFile(suffix=".gz", delete=True) as tmp:
        tmp_path = Path(tmp.name)
        # Stream download
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.writelines(resp.iter_bytes(chunk_size=8192))

        # Process
        with gzip.open(tmp_path, "rt") as gf:
            for line in gf:
                if not line.strip():
                    continue
                total += 1
                try:
                    record: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if is_2025_paper(record) and matches_nlp_venue(record):
                    cid = record.get("corpusid")
                    if cid and cid not in existing_nlp_ids:
                        nlp_2025_new += 1
                        new_ids.add(cid)
                    elif cid:
                        nlp_2025_updated += 1

    return total, nlp_2025_new, nlp_2025_updated, new_ids


def load_existing_ids(path: Path) -> set[int]:
    """Load existing corpus IDs from the current dataset."""
    if not path.exists():
        return set()

    with open(path) as f:
        return {int(stripped) for raw_line in f if (stripped := raw_line.strip())}


def main() -> None:
    """Investigate differences between S2 dataset releases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and process diff files (slow, requires API key)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Only download first N files per diff step (0=all)",
    )
    parser.add_argument(
        "--existing-ids",
        type=Path,
        default=Path("data/s2orc-nlp/corpus_ids.txt"),
        help="Path to existing corpus_ids.txt",
    )
    args = parser.parse_args()

    timeout = httpx.Timeout(60.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        # Step 1: List releases
        logger.info("=== Step 1: Enumerate releases ===")
        releases = list_releases(client)
        relevant = [r for r in releases if BASE_RELEASE <= r <= TARGET_RELEASE]
        logger.info(f"All releases: {len(releases)}")
        logger.info(f"Releases {BASE_RELEASE} to {TARGET_RELEASE}: {relevant}")

        # Step 2: Fetch diff metadata
        logger.info(
            f"\n=== Step 2: Fetch diffs ({BASE_RELEASE} → {TARGET_RELEASE}) ==="
        )
        diffs = get_diffs(client, BASE_RELEASE, TARGET_RELEASE, "papers")

        # Normalise response: API returns {dataset, start_release, end_release, diffs: [...]}
        diff_list: list[dict[str, Any]] = []
        if isinstance(diffs, dict):
            logger.info(f"Response keys: {list(diffs.keys())}")
            logger.info(f"  dataset: {diffs.get('dataset')}")
            logger.info(f"  start_release: {diffs.get('start_release')}")
            logger.info(f"  end_release: {diffs.get('end_release')}")
            diff_list: list[dict[str, Any]] = diffs.get("diffs", [])
        else:
            diff_list = diffs

        logger.info(f"Number of diff steps: {len(diff_list)}")
        total_update_files = 0
        total_delete_files = 0
        for i, entry in enumerate(diff_list):
            update_files = entry.get("update_files", [])
            delete_files = entry.get("delete_files", [])
            total_update_files += len(update_files)
            total_delete_files += len(delete_files)
            logger.info(
                f"  Diff {i} ({entry.get('from_release', '?')} → {entry.get('to_release', '?')}): "
                f"{len(update_files)} update files, {len(delete_files)} delete files"
            )
            if update_files:
                logger.info(f"    First update URL: {update_files[0][:100]}...")
        logger.info(
            f"Total: {total_update_files} update files, {total_delete_files} delete files"
        )

        if not args.download:
            logger.info(
                "\nRun with --download to actually process diff files and count 2025 papers."
            )
            return

        # Step 3: Download and count
        logger.info("\n=== Step 3: Download diffs and count 2025 NLP papers ===")
        existing_ids = load_existing_ids(args.existing_ids)
        logger.info(f"Existing corpus IDs: {len(existing_ids):,}")

        total_new_2025 = 0
        total_updated_2025 = 0
        all_new_ids: set[int] = set()
        files_sampled = 0
        files_total = 0

        for diff_idx, diff_entry in enumerate(diff_list):
            update_files: list[str] = diff_entry.get("update_files", [])
            if not update_files:
                continue

            files_total += len(update_files)
            files_to_process = (
                update_files[: args.sample] if args.sample else update_files
            )
            logger.info(
                f"\nProcessing diff {diff_idx}: "
                f"{len(files_to_process)}/{len(update_files)} update files"
            )

            for file_idx, url in enumerate(files_to_process):
                logger.info(
                    f"  File {file_idx + 1}/{len(files_to_process)}: downloading..."
                )
                total, new, updated, new_ids = download_and_count_diff_file(
                    client, url, existing_ids | all_new_ids
                )
                all_new_ids |= new_ids
                total_new_2025 += new
                total_updated_2025 += updated
                files_sampled += 1
                logger.info(
                    f"    Records: {total:,}, "
                    f"New 2025 NLP: {new:,}, "
                    f"Updated 2025 NLP: {updated:,}"
                )

        logger.info("\n=== Results ===")
        logger.info(f"Files processed: {files_sampled}/{files_total}")
        logger.info(f"New 2025 NLP papers found: {total_new_2025:,}")
        logger.info(f"Updated 2025 NLP papers: {total_updated_2025:,}")
        logger.info(f"New unique corpus IDs: {len(all_new_ids):,}")

        # Extrapolate if sampling
        if args.sample and files_sampled < files_total:
            scale = files_total / files_sampled
            extrapolated = int(len(all_new_ids) * scale)
            logger.info(
                f"\nExtrapolation ({files_sampled} → {files_total} files, "
                f"{scale:.1f}x):"
            )
            logger.info(f"  Estimated new 2025 NLP from diffs: ~{extrapolated:,}")
            estimated_total = extrapolated + 398
        else:
            estimated_total = len(all_new_ids) + 398

        logger.info(f"\nEstimated total 2025 NLP: {estimated_total:,}")
        logger.info("Target (from S2 search API): ~7,224")

        coverage = estimated_total / 7224 * 100
        logger.info(f"Estimated coverage: {coverage:.1f}%")

        if coverage >= 80:
            logger.info(
                "\n→ VERDICT: Diffs provide sufficient coverage. "
                "Proceed with diff-based approach."
            )
        else:
            logger.info(
                "\n→ VERDICT: Diffs insufficient. "
                "Fall back to full pipeline into data/s2orc-nlp-v2/."
            )


if __name__ == "__main__":
    main()
