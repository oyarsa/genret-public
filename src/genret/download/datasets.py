"""List and download Semantic Scholar datasets (standalone version)."""

import asyncio
import json
from pathlib import Path
from typing import Annotated

import httpx
import typer

from genret.utils import (
    ProgressManager,
    die,
    download_file,
    get_filename_from_url,
    make_app,
    require_env,
)

MAX_CONCURRENT_DOWNLOADS = 10
"""Maximum number of concurrent dataset file downloads."""

app = make_app("List and download Semantic Scholar datasets")


@app.command(name="list")
def list_datasets(
    show_json: Annotated[
        bool,
        typer.Option(
            "--json", help="Output data in JSON format instead of plain text."
        ),
    ] = False,
) -> None:
    """List available datasets."""
    asyncio.run(_list_datasets(show_json))


async def _list_datasets(show_json: bool) -> None:
    """List available datasets from the Semantic Scholar API.

    Args:
        show_json: If True, print raw JSON; otherwise print name and description.
    """
    async with httpx.AsyncClient() as client:
        # Get latest release ID
        response = await client.get(
            "https://api.semanticscholar.org/datasets/v1/release/latest"
        )
        response.raise_for_status()
        releases = response.json()

        release_id = releases["release_id"]
        print(f"Latest release ID: {release_id}")

        # Get datasets for the latest release
        response = await client.get(
            f"https://api.semanticscholar.org/datasets/v1/release/{release_id}"
        )
        response.raise_for_status()
        data = response.json()

    if show_json:
        print(json.dumps(data["datasets"], indent=2))
    else:
        for dataset in data["datasets"]:
            print(dataset["name"], dataset["description"].strip(), sep="\n", end="\n\n")


@app.command(name="download", help="Download a dataset.", no_args_is_help=True)
def download_dataset(
    dataset_name: Annotated[
        str, typer.Argument(help="Name of the dataset to download.")
    ],
    output_path: Annotated[
        Path, typer.Argument(help="Directory to save the downloaded files.")
    ],
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            help="Limit the number of files to download. Useful for testing.",
        ),
    ] = None,
) -> None:
    """Download dataset files from the Semantic Scholar API to the output path."""
    api_key = require_env("SEMANTIC_SCHOLAR_API_KEY")

    output_path.mkdir(parents=True, exist_ok=True)
    asyncio.run(_download(dataset_name, output_path, api_key, limit))


async def _download(
    dataset_name: str, output_path: Path, api_key: str, limit: int | None
) -> None:
    """Download dataset files from the Semantic Scholar API to the output path.

    Args:
        dataset_name: Name of the dataset to download (e.g. ``"papers"``).
        output_path: Directory to save downloaded files and dataset metadata.
        api_key: Semantic Scholar API key for authenticated download requests.
        limit: Maximum number of files to download, or None for all files.
    """
    async with httpx.AsyncClient() as client:
        # Get latest release's ID
        response = await client.get(
            "https://api.semanticscholar.org/datasets/v1/release/latest"
        )
        response.raise_for_status()
        release_id = response.json()["release_id"]
        print(f"Latest release ID: {release_id}")

        # Get the download links for the dataset
        response = await client.get(
            f"https://api.semanticscholar.org/datasets/v1/release/{release_id}/dataset/{dataset_name}/",
            headers={"x-api-key": api_key},
        )
        response.raise_for_status()
        dataset = response.json()

        # Save dataset metadata
        (output_path / "dataset.json").write_text(json.dumps(dataset, indent=2))

        if "files" not in dataset or not dataset["files"]:
            die("No files found.")

        urls = dataset["files"][:limit]
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        print(f"Files to download: {len(urls)}")

        async def download_worker(url: str, progress: ProgressManager) -> None:
            async with semaphore:
                file_name = get_filename_from_url(url)
                file_path = output_path / file_name
                file_task = progress.add_file(file_name)
                try:
                    await download_file(
                        url,
                        file_path,
                        on_progress=lambda b, t: progress.update_file(file_task, b, t),
                    )
                    progress.advance()
                except Exception:
                    progress.advance(failed=True)
                    raise
                finally:
                    progress.remove_file(file_task)

        with ProgressManager(len(urls), "Downloading") as progress:
            tasks = [
                asyncio.create_task(download_worker(url, progress)) for url in urls
            ]
            await asyncio.gather(*tasks)
        print("Download complete!")


if __name__ == "__main__":
    app()
