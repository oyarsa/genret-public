"""Shared utility functions and async utilities for downloads and progress tracking."""

import asyncio
import functools
import io
import json
import logging
import operator
import os
import random
import sys
import urllib.parse
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Literal, NoReturn, Self, overload, override

import aiofiles
import httpx
import typer
from pydantic import BaseModel
from rich.console import RenderableType
from rich.filesize import decimal as filesize
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from genret._shed.aiofiles import AsyncWritable as _AsyncWritable
from genret._shed.aiofiles import as_async_writable
from genret._shed.compression import open_gz, open_zstd
from genret.log_config import console
from genret.types import CorpusId

logger = logging.getLogger(__name__)


type OpenMode = Literal["a", "w"]
"""File open mode for writing. "a": append after existing, "w": write (overwrite existing)."""

type JsonlMode = Literal["r", "a", "w"]
"""File open mode for JSONL files. "r": read, "a": append, "w": write (overwrite)."""


def make_app(help_text: str) -> typer.Typer:
    """Create a Typer app with standard project configuration."""
    return typer.Typer(
        help=help_text,
        context_settings={"help_option_names": ["-h", "--help"]},
        add_completion=False,
        rich_markup_mode="rich",
        pretty_exceptions_show_locals=False,
        no_args_is_help=True,
    )


def resolve_jsonl(path: Path) -> Path:
    """Find the actual JSONL file on disk, preferring `.jsonl.zst` over `.jsonl.gz`.

    Given any variant of a path (plain, ``.jsonl.gz``, or ``.jsonl.zst``), resolves
    to the canonical ``.jsonl`` base and then returns the first existing variant
    in preference order: ``.jsonl.zst`` > ``.jsonl.gz`` > ``.jsonl``.

    When no variant exists (new file), returns the ``.jsonl.zst`` path so that
    new output files use zstd compression by default.

    Args:
        path: Candidate JSONL file path (any compression suffix or none).

    Returns:
        The path variant that exists on disk, or the ``.jsonl.zst`` path if none do.
    """
    name = path.name
    if name.endswith(".jsonl.gz"):
        base = path.with_name(name.removesuffix(".gz"))
    elif name.endswith(".jsonl.zst"):
        base = path.with_name(name.removesuffix(".zst"))
    else:
        base = path

    zst = base.with_name(base.name + ".zst")
    gz = base.with_name(base.name + ".gz")

    if zst.exists():
        return zst
    if gz.exists():
        return gz
    if base.exists():
        return base
    return zst


@contextmanager
def open_auto(path: Path, mode: JsonlMode) -> Generator[IO[str]]:
    """Open a file for reading or writing, with transparent gzip/zstd support.

    When `path` ends with `.gz`, uses :func:`gzip.open` in text mode; when it ends
    with `.zst`, uses :func:`compression.zstd.open` in text mode; otherwise falls
    back to plain :func:`open`.

    Args:
        path: File path (plain, `.gz`, or `.zst`).
        mode: `"r"` to read, `"w"` to write (overwrite), or `"a"` to append.

    Yields:
        A text-mode file handle.

    Raises:
        Whatever the underlying open call can raise.
    """
    z_mode: Literal["rt", "wt", "at"] = (
        "rt" if mode == "r" else "wt" if mode == "w" else "at"
    )
    if path.name.endswith(".gz"):
        with open_gz(path, z_mode) as f:
            yield f
    elif path.name.endswith(".zst"):
        with open_zstd(path, z_mode) as f:
            yield f
    else:
        with open(path, mode, encoding="utf-8") as f:
            yield f


def glob_jsonl(directory: Path) -> list[Path]:
    """Glob for JSONL files in `directory`, including all compressed variants.

    Returns paths matching ``*.jsonl``, ``*.jsonl.gz``, and ``*.jsonl.zst``, sorted by
    name.

    Args:
        directory: Directory to search.

    Returns:
        Sorted list of matching paths, deduplicated by base name (`.zst` beats `.gz`
        beats plain `.jsonl` when multiple variants co-exist).
    """
    # Build a dict keyed by the base `*.jsonl` name, processing in increasing
    # priority order so that higher-priority variants overwrite lower ones.
    candidates: dict[str, Path] = {}
    for ext in (".jsonl", ".jsonl.gz", ".jsonl.zst"):
        for path in directory.glob(f"*{ext}"):
            base = path.name.removesuffix(ext) + ".jsonl"
            candidates[base] = path
    return sorted(candidates.values(), key=lambda p: p.name)


def count_lines(path: Path) -> int:
    """Count the number of lines in a text file.

    Handles gzip-compressed files transparently.

    Args:
        path: Path to the file (plain or ``.gz``).

    Returns:
        Total line count.
    """
    with open_auto(path, "r") as f:
        return countit(f)


class _JsonlFile:
    """JSONL file iterable that lazily counts lines for progress bar integration.

    Exposes ``__length_hint__`` so that :func:`track_progress` can show a determinate
    progress bar without callers passing `total` explicitly. The line count is computed
    lazily on first access, so non-progress call sites pay no overhead.
    """

    def __init__(self, path: Path, *, count: bool = True) -> None:
        """Initialise from `path`, with optional lazy line counting."""
        self._path = path
        self._count = count

    @functools.cached_property
    def _hint(self) -> int:
        """Line count, computed lazily on first access."""
        return count_lines(self._path) if self._count else 0

    def __length_hint__(self) -> int:
        """Return the total line count as a size hint."""
        return self._hint

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield parsed JSON objects from each non-empty line."""
        with open_auto(self._path, "r") as f:
            yield from _iter_jsonl_handle(f)


def _iter_jsonl_handle(f: IO[str] | io.TextIOBase) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from an open text handle, skipping empty lines."""
    for line in f:
        if line.strip():
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("JSONL line is not an object")
            yield obj


def iter_jsonl(
    source: Path | IO[str], *, count: bool = True
) -> Iterable[dict[str, Any]]:
    """Iterate over records from a JSONL file, skipping empty lines.

    When `source` is a :class:`~pathlib.Path`, the returned iterable exposes
    ``__length_hint__`` so that :func:`track_progress` can display a
    determinate progress bar.  The line count is computed lazily — only when
    ``__length_hint__`` is actually called — so non-progress call sites pay
    no overhead.

    Automatically handles gzip-compressed files if the path ends with .gz.

    Args:
        source: Path to the JSONL file (plain or .gz), or an open text file object.
        count: Enable lazy line counting for ``__length_hint__``.  Set to
            ``False`` to suppress counting entirely.  Ignored when `source`
            is a file handle.

    Returns:
        An iterable of parsed JSON objects from each non-empty line.

    Raises:
        json.JSONDecodeError if a line is not valid JSON.
        ValueError if the parsed line is not an object.
    """
    if isinstance(source, Path):
        return _JsonlFile(source, count=count)
    return _iter_jsonl_handle(source)


class CallbackColumn(ProgressColumn):
    """A progress column that displays text from a callback.

    The callback is invoked on each render and should return the text to display.
    """

    def __init__(self, callback: Callable[[], str]) -> None:
        super().__init__()
        self._callback = callback

    @override
    def render(self, task: Task) -> Text:
        return Text(self._callback())


@overload
def track_progress[T](
    iterable: AsyncIterator[T],
    desc: str = ...,
    total: int | None = ...,
    extra_columns: Sequence[ProgressColumn] = ...,
) -> AsyncIterator[T]: ...


@overload
def track_progress[T](
    iterable: Iterable[T],
    desc: str = ...,
    total: int | None = ...,
    extra_columns: Sequence[ProgressColumn] = ...,
) -> Iterable[T]: ...


def track_progress[T](
    iterable: AsyncIterator[T] | Iterable[T],
    desc: str = "Processing...",
    total: int | None = None,
    extra_columns: Sequence[ProgressColumn] = (),
) -> AsyncIterator[T] | Iterable[T]:
    """Wrap an iterable or async iterator with a Rich progress bar.

    When `total` is not provided, attempts `operator.length_hint` to
    infer the length automatically.

    Args:
        iterable: The iterable or async iterator to track.
        desc: Description shown next to the progress bar.
        total: Total number of items.  None tries length_hint, then shows a pulsing bar.
        extra_columns: Additional Rich columns appended after the default ones.

    Returns:
        An iterable or async iterator that yields the same items with progress display.
    """
    if total is None:
        hint = operator.length_hint(iterable)
        if hint > 0:
            total = hint

    if isinstance(iterable, AsyncIterator):
        return _async_track(iterable, desc, total, extra_columns)

    return _sync_track(iterable, desc, total, extra_columns)


def _make_progress(
    extra_columns: Sequence[ProgressColumn] = (),
) -> Progress:
    """Create a Progress bar with the standard column layout.

    Auto-refresh is disabled; callers should call `progress.refresh()` manually after
    each update to avoid unnecessary repaints.

    Args:
        extra_columns: Additional columns appended after the default ones.

    Returns:
        A configured Progress instance.
    """
    return Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        *extra_columns,
        console=console,
        auto_refresh=False,
    )


def _sync_track[T](
    iterable: Iterable[T],
    desc: str,
    total: int | None,
    extra_columns: Sequence[ProgressColumn] = (),
) -> Iterable[T]:
    """Wrap a sync iterable with a Rich progress bar, refreshing only on change."""
    with _make_progress(extra_columns) as progress:
        task_id = progress.add_task(desc, total=total)

        for item in iterable:
            yield item
            progress.advance(task_id)
            progress.refresh()


async def _async_track[T](
    iterable: AsyncIterator[T],
    desc: str,
    total: int | None,
    extra_columns: Sequence[ProgressColumn] = (),
) -> AsyncIterator[T]:
    """Wrap an async iterator with a Rich progress bar, refreshing only on change."""
    with _make_progress(extra_columns) as progress:
        task_id = progress.add_task(desc, total=total)

        async for item in iterable:
            yield item
            progress.advance(task_id)
            progress.refresh()


# === Constants ===

DOWNLOAD_TIMEOUT = 3600
"""HTTP timeout in seconds for file downloads (1 hour)."""
MAX_RETRIES = 3
"""Maximum number of download retry attempts before failing."""
RETRY_DELAY = 5
"""Seconds to wait between download retries."""

AUTH_ERROR_CODES = {400, 401, 403}
"""HTTP status codes that indicate expired or invalid pre-signed URLs."""
RATE_LIMIT_CODE = 429
"""HTTP status code for rate-limiting responses."""
RATE_LIMIT_BASE_DELAY = 60
"""Base delay in seconds for exponential backoff on rate-limit responses."""

# Type alias for URL refresher callback
UrlRefresher = Callable[[], Awaitable[str]]


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, rate: float) -> None:
        """Initialise with the given request rate.

        Args:
            rate: Maximum requests per second.
        """
        self.rate = rate
        self.interval = 1.0 / rate
        self.lock = asyncio.Lock()
        self.last_request = 0.0

    async def acquire(self) -> None:
        """Wait until a request can be made within rate limit."""
        async with self.lock:
            now = asyncio.get_running_loop().time()
            wait_time = self.last_request + self.interval - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_request = asyncio.get_running_loop().time()


def die(message: str) -> NoReturn:
    """Log an error message and exit."""
    logger.error(message)
    sys.exit(1)


def confirm_force() -> None:
    """Prompt for interactive confirmation when `--force` is used.

    Prints a y/N prompt to stderr and reads from stdin. Defaults to `no` —
    any answer other than literal `y` (case-insensitive) aborts with exit code 1.
    """
    try:
        answer = input("--force will overwrite existing data. Continue? [y/N] ")
    except EOFError:
        answer = ""

    if answer.strip().lower() != "y":
        sys.exit(1)


def require_env(name: str) -> str:
    """Get required environment variable value or exit with an error."""
    value = os.environ.get(name)
    if not value:
        die(f"{name} environment variable is required.")
    return value


def backup_files(files: list[Path], backup_suffix: str = ".bak") -> None:
    """Back up files before replacing/deleting them.

    If a backup path already exists, appends a UTC timestamp to avoid collisions.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    for path in files:
        if not path.exists():
            continue

        backup_path = path.with_suffix(path.suffix + backup_suffix)
        if backup_path.exists():
            backup_path = path.with_suffix(f"{path.suffix}.{timestamp}{backup_suffix}")

        path.rename(backup_path)
        logger.info("  Backed up: %s -> %s", path.name, backup_path.name)


class _FailedColumn(ProgressColumn):
    """Custom column that shows failed count only for main task."""

    @override
    def render(self, task: Task) -> Text:
        """Render the failed count."""
        if task.fields.get("is_main"):
            failed = task.fields.get("failed", 0)
            return Text(f"({failed} failed)", style="red")
        return Text("")


class _FileSizeColumn(ProgressColumn):
    """Custom column that shows download size only for file tasks (not processing)."""

    @override
    def render(self, task: Task) -> Text:
        """Render the download size."""
        if (
            not task.fields.get("is_main")
            and not task.fields.get("is_processing")
            and task.total is not None
        ):
            return Text(f"{filesize(int(task.completed))}/{filesize(int(task.total))}")
        return Text("")


class _FileBarColumn(ProgressColumn):
    """Bar column that hides for processing tasks."""

    def __init__(self) -> None:
        """Initialise with an inner BarColumn."""
        super().__init__()
        self._inner = BarColumn()

    @override
    def render(self, task: Task) -> RenderableType:
        """Render the bar, but hide for processing tasks."""
        if task.fields.get("is_processing"):
            return Text("")
        return self._inner.render(task)


class _FileTimeColumn(ProgressColumn):
    """Time elapsed column that hides for processing tasks."""

    def __init__(self) -> None:
        """Initialise with an inner TimeElapsedColumn."""
        super().__init__()
        self._inner = TimeElapsedColumn()

    @override
    def render(self, task: Task) -> RenderableType:
        """Render time elapsed, but hide for processing tasks."""
        if task.fields.get("is_processing"):
            return Text("")
        return self._inner.render(task)


class _MofNColumn(MofNCompleteColumn):
    """M/N column that hides for file tasks (shows only for main task)."""

    @override
    def render(self, task: Task) -> Text:
        """Render M/N, but only for main task."""
        if not task.fields.get("is_main"):
            return Text("")
        return super().render(task)


class _TimeRemainingColumn(TimeRemainingColumn):
    """Time remaining column that hides for file/processing tasks."""

    @override
    def render(self, task: Task) -> Text:
        """Render time remaining, but only for main task."""
        if not task.fields.get("is_main"):
            return Text("")
        return super().render(task)


class ProgressManager:
    """Manages progress display for concurrent operations using rich.

    Supports both a global progress bar and per-file progress bars for downloads.
    Use `logger.info()` for messages — it won't interfere with the display.

    Usage:

    ```py
    with ProgressManager(total=100, description="Downloading") as progress:
        file_task = progress.add_file("file.txt", total_bytes=1000)
        for chunk in chunks:
            progress.update_file(file_task, len(chunk))
        progress.remove_file(file_task)
        progress.advance()
    ```
    """

    def __init__(
        self,
        total: int,
        description: str = "Processing",
    ) -> None:
        """Initialise the progress manager.

        Args:
            total: Total number of items to process.
            description: Description shown in progress bar.
        """
        self.total = total
        self.description = description
        self._progress: Progress | None = None
        self._main_task_id: TaskID | None = None
        self._failed_count = 0
        self._completed_count = 0

    def __enter__(self) -> Self:
        """Start progress display."""
        # Progress columns - some render only for main task, some only for file tasks
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            _FileBarColumn(),  # Hides for processing tasks
            _MofNColumn(),  # Shows X/Y only for main task
            _FileSizeColumn(),  # Shows bytes only for file download tasks
            _FailedColumn(),  # Shows failed count for main task
            _FileTimeColumn(),  # Hides for processing tasks
            _TimeRemainingColumn(),  # Shows only for main task
            console=console,
            expand=False,
        )
        self._progress.__enter__()
        self._main_task_id = self._progress.add_task(
            self.description, total=self.total, failed=0, is_main=True
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop progress display."""
        if self._progress:
            self._progress.__exit__(exc_type, exc_val, exc_tb)

    def add_file(self, filename: str, total_bytes: int | None = None) -> TaskID:
        """Add a progress bar for a file download.

        Args:
            filename: Name to display for this file.
            total_bytes: Total size in bytes (None if unknown).

        Returns:
            TaskID to use with update_file() and remove_file().
        """
        if not self._progress:
            raise RuntimeError("ProgressManager not entered")

        # Create a file-specific progress bar
        return self._progress.add_task(
            f"  ↳ {filename[:40]}",
            total=total_bytes,
            visible=True,
        )

    def update_file(
        self, task_id: TaskID, bytes_downloaded: int, total_bytes: int | None = None
    ) -> None:
        """Update a file's download progress.

        Args:
            task_id: The task ID from add_file().
            bytes_downloaded: Total bytes downloaded so far.
            total_bytes: Total file size (updates the task total if provided).
        """
        if not self._progress:
            return

        if total_bytes is not None:
            self._progress.update(
                task_id, completed=bytes_downloaded, total=total_bytes
            )
        else:
            self._progress.update(task_id, completed=bytes_downloaded)

    def file_processing(self, task_id: TaskID, filename: str) -> None:
        """Mark a file as processing (download complete, now processing).

        Args:
            task_id: The task ID from add_file().
            filename: Filename to show in the processing message.
        """
        if self._progress:
            self._progress.update(
                task_id,
                description=f"  ↳ Processing {filename[:40]}...",
                completed=0,
                total=None,  # Indeterminate progress
                is_processing=True,
            )

    def remove_file(self, task_id: TaskID) -> None:
        """Remove a file's progress bar.

        Args:
            task_id: The task ID from add_file().
        """
        if self._progress:
            self._progress.remove_task(task_id)

    def advance(self, failed: bool = False) -> None:
        """Advance the main progress by one item.

        Args:
            failed: If True, increment the failed counter.
        """
        if self._progress and self._main_task_id is not None:
            if failed:
                self._failed_count += 1
            self._completed_count += 1
            self._progress.update(
                self._main_task_id, advance=1, failed=self._failed_count
            )

    @property
    def failed_count(self) -> int:
        """Get the number of failed items."""
        return self._failed_count


# === File-based ID utilities ===


def load_references(
    path: Path, *, show_progress: bool = False
) -> dict[CorpusId, list[CorpusId]]:
    """Load references from JSONL file.

    Args:
        path: Path to citations/references JSONL file.
        show_progress: Whether to show a progress bar.
    """
    if not path.exists():
        return {}

    source: Iterable[dict[str, Any]] = iter_jsonl(path)
    if show_progress:
        source = track_progress(source, desc="Loading references...")

    return {
        CorpusId(record["paper_id"]): [CorpusId(r) for r in record["references"]]
        for record in source
    }


def load_ids_from_file(path: Path) -> set[CorpusId]:
    """Load integer IDs from a text file (one per line)."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {CorpusId(int(line)) for line_ in f if (line := line_.strip())}


def load_processed_strings(path: Path) -> set[str]:
    """Load set of processed string keys from a text file (one per line).

    This is the synchronous counterpart of :func:`load_processed_files`,
    useful when loading progress outside an async context.

    Args:
        path: Path to the progress/tracking file.

    Returns:
        Set of non-empty stripped lines.
    """
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {line for line_ in f if (line := line_.strip())}


# === Async file-tracking utilities ===


async def load_processed_files(path: Path) -> set[str]:
    """Load set of processed filenames/IDs from a text file (one per line)."""
    if not path.exists():
        return set()
    async with aiofiles.open(path) as f:
        content = await f.read()
        return {line for line_ in content.splitlines() if (line := line_.strip())}


async def append_processed_file(path: Path, filename: str) -> None:
    """Append an entry to a processed-files tracking file."""
    async with aiofiles.open(path, "a") as f:
        await f.write(f"{filename}\n")


class _SyncFileWrapper:
    """Thin adapter that wraps a synchronous file handle with async-compatible methods.

    Intended for gzip files used inside :class:`AsyncLineWriter`.  All writes
    happen under the writer's asyncio lock, so concurrent safety is preserved
    without a thread pool.
    """

    def __init__(self, file: IO[str] | io.TextIOBase) -> None:
        """Wrap *file* for use in async contexts."""
        self._file = file

    async def write(self, data: str) -> int:
        """Write *data* synchronously and return the number of characters written."""
        return self._file.write(data)

    async def flush(self) -> None:
        """Flush the underlying file."""
        self._file.flush()


class AsyncLineWriter:
    """Async line writer with internal locking for concurrent coroutine safety.

    Construct with `AsyncLineWriter.open`.

    Writes individual lines (with trailing newline) to a file, serialising access via
    an asyncio lock so that multiple coroutines can safely call `write_line`
    concurrently.
    """

    _file: _AsyncWritable
    _lock: asyncio.Lock

    def __init__(self, file: _AsyncWritable) -> None:
        """Initialise the writer. Construct one with AsyncLineWriter.open."""
        self._lock = asyncio.Lock()
        self._file = file

    @classmethod
    @asynccontextmanager
    async def open(cls, path: Path, mode: OpenMode = "a") -> AsyncGenerator[Self]:
        """Open an async file and yield the writer.

        For ``.gz`` and ``.zst`` paths, uses synchronous compressed open wrapped
        in :class:`_SyncFileWrapper` (line-by-line writes are fast enough to run
        under the existing asyncio lock). For plain paths, uses :mod:`aiofiles`.

        Args:
            path: Destination file path (plain, ``.gz``, or ``.zst``).
            mode: File open mode — `"a"`` to append, `"w"` to overwrite.
        """
        z_mode: Literal["wt", "at"] = "wt" if mode == "w" else "at"
        if path.name.endswith(".gz"):
            with open_gz(path, z_mode) as f:
                yield cls(_SyncFileWrapper(f))
        elif path.name.endswith(".zst"):
            with open_zstd(path, z_mode) as f:
                yield cls(_SyncFileWrapper(f))
        else:
            async with aiofiles.open(path, mode, encoding="utf-8") as f:
                yield cls(as_async_writable(f))

    async def write_line(self, line: str, *, flush: bool = False) -> None:
        r"""Write `line` plus a newline, all under lock.

        `line` is always written with exactly one trailing "\n". Any existing newlines
        are stripped first.

        Args:
            line: The text to write.
            flush: Whether to flush after writing.
        """
        async with self._lock:
            await self._file.write(line.rstrip("\r\n") + "\n")
            if flush:
                await self._file.flush()

    async def write_obj(self, obj: BaseModel, *, flush: bool = False) -> None:
        """Serialise a Pydantic model to JSON and write it as a line.

        Args:
            obj: The Pydantic model to serialise and write.
            flush: Whether to flush after writing.
        """
        await self.write_line(obj.model_dump_json(), flush=flush)

    async def write_lines(self, lines: Iterable[str], *, flush: bool = True) -> None:
        r"""Write all lines to file, with exactly one "\n" between them, all under lock.

        Args:
            lines: The text to write.
            flush: Whether to flush after writing all lines.
        """
        for line in lines:
            await self.write_line(line)

        if flush:
            await self.flush()

    async def write_objs(
        self, objs: Iterable[BaseModel], *, flush: bool = True
    ) -> None:
        """Serialise an iterable of Pydantic models to JSONLines format, all under lock.

        Args:
            objs: The Pydantic models to serialise and write.
            flush: Whether to flush after writing all objects.
        """
        for obj in objs:
            await self.write_line(obj.model_dump_json())

        if flush:
            await self.flush()

    async def flush(self) -> None:
        """Flush buffer to file."""
        await self._file.flush()


class LineWriter:
    """Synchronous line writer for JSONL and other line-oriented files.

    Construct with `LineWriter.open`.

    Context manager that writes individual lines (with trailing newline) to a
    file.  Mirrors `AsyncLineWriter` but for synchronous code.
    """

    def __init__(self, file: IO[str] | io.TextIOBase) -> None:
        """Initialise the writer. Construct one with `LineWriter.open`."""
        self._file = file

    @classmethod
    @contextmanager
    def open(cls, path: Path, mode: OpenMode = "a") -> Generator[Self]:
        """Open a file and yield the writer.

        Transparently handles ``.gz`` paths via :func:`open_jsonl`.

        Args:
            path: Destination file path (plain or ``.gz``).
            mode: File open mode — `"a"` to append, `"w"` to overwrite.
        """
        with open_auto(path, mode) as f:
            yield cls(f)

    def write_line(self, line: str, *, flush: bool = False) -> None:
        r"""Write `line` plus a newline.

        `line` is always written with exactly one trailing "\n". Any existing newlines
        are stripped first.

        Args:
            line: The text to write.
            flush: Whether to flush after writing.
        """
        self._file.write(line.rstrip("\r\n") + "\n")
        if flush:
            self.flush()

    def write_obj(self, obj: BaseModel, *, flush: bool = False) -> None:
        """Serialise a Pydantic model to JSON and write it as a line.

        Args:
            obj: The Pydantic model to serialise and write.
            flush: Whether to flush after writing.
        """
        self.write_line(obj.model_dump_json(), flush=flush)

    def write_lines(self, lines: Iterable[str], *, flush: bool = True) -> None:
        r"""Write all lines to file, with exactly one "\n" between them.

        Args:
            lines: The text to write.
            flush: Whether to flush after writing all lines.
        """
        for line in lines:
            self.write_line(line)

        if flush:
            self.flush()

    def write_objs(self, objs: Iterable[BaseModel], *, flush: bool = True) -> None:
        """Serialise an iterable of Pydantic models to JSONLines format.

        Args:
            objs: The Pydantic models to serialise and write.
            flush: Whether to flush after writing all objects.
        """
        for obj in objs:
            self.write_line(obj.model_dump_json())

        if flush:
            self.flush()

    def flush(self) -> None:
        """Flush buffer to file."""
        self._file.flush()


# === Download utilities ===


def get_filename_from_url(url: str) -> str:
    """Extract filename from URL."""
    return urllib.parse.urlparse(url).path.split("/")[-1]


# Type alias for progress callback (receives bytes downloaded and total size)
ProgressCallback = Callable[[int, int | None], None]


async def download_file(
    url: str,
    dest: Path,
    url_refresher: UrlRefresher | None = None,
    on_progress: ProgressCallback | None = None,
) -> int:
    """Download a file with retries. Returns bytes downloaded.

    Args:
        url: The URL to download from.
        dest: Destination path for the downloaded file.
        url_refresher: Optional async callback to get a fresh URL when auth errors occur.
            This is useful for pre-signed S3 URLs that may expire during long runs.
        on_progress: Optional callback called with bytes downloaded so far.
    """
    part_path = dest.with_suffix(dest.suffix + ".part")
    current_url = url
    rate_limit_backoff = 0  # Exponential backoff counter for 429s

    for attempt in range(MAX_RETRIES):
        try:
            bytes_downloaded = await _try_download_file(
                current_url, part_path, on_progress
            )
            part_path.rename(dest)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code

            # Handle auth errors by refreshing URL
            if status in AUTH_ERROR_CODES and url_refresher:
                logger.warning("Auth error for %s, refreshing URL...", dest.name)
                current_url = await url_refresher()
                continue  # Don't count as retry

            # Handle rate limiting with exponential backoff
            if status == RATE_LIMIT_CODE:
                delay = RATE_LIMIT_BASE_DELAY * (2**rate_limit_backoff)
                rate_limit_backoff += 1
                logger.warning("Rate limited for %s, waiting %ds...", dest.name, delay)
                await asyncio.sleep(delay)
                continue  # Don't count as retry

            logger.warning(
                "Error downloading %s: %s. Retrying... (Attempt %d/%d)",
                dest.name,
                e,
                attempt + 1,
                MAX_RETRIES,
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            logger.warning(
                "Error downloading %s: %s. Retrying... (Attempt %d/%d)",
                dest.name,
                e,
                attempt + 1,
                MAX_RETRIES,
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
        else:
            return bytes_downloaded

    logger.error("Failed to download %s after %d attempts.", dest.name, MAX_RETRIES)
    raise RuntimeError(f"Download failed after {MAX_RETRIES} attempts")


async def _try_download_file(
    url: str, part_path: Path, on_progress: ProgressCallback | None = None
) -> int:
    """Download the file in chunks. Returns bytes downloaded."""
    async with (
        httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0)) or None
        bytes_written = 0
        async with aiofiles.open(part_path, "wb") as f:
            async for chunk in response.aiter_bytes(chunk_size=8192):
                await f.write(chunk)
                bytes_written += len(chunk)
                if on_progress:
                    on_progress(bytes_written, total_size)
        return bytes_written


def load_split_ids(splits_dir: Path) -> set[CorpusId]:
    """Load corpus IDs from train/dev/test.txt in splits directory."""
    ids: set[CorpusId] = set()
    for name in ["train", "dev", "test"]:
        ids |= load_ids_from_file(splits_dir / f"{name}.txt")
    return ids


def unique[T](it: Iterable[T]) -> list[T]:
    """Get unique values from `it`, preserving the original order."""
    return list(dict.fromkeys(it))


def countit[T](it: Iterable[T]) -> int:
    """Count number of items in `it`. Exhausts the iterator."""
    return sum(1 for _ in it)


def safediv(a: float | int, b: float | int) -> float:  # noqa: PYI041
    """Calculate `a/b` if b > 0, else returns 0."""
    if b > 0:
        return a / b
    return 0.0


class Unreachable(Exception):  # noqa: N818
    """Unreachable piece of code."""

    def __init__(self) -> None:
        super().__init__(self, "Unreachable! If you're seeing this, it's a bug.")


def sample[T](rng: random.Random, population: Sequence[T], k: int | None) -> list[T]:
    """Sample `k` elements from from `population`.

    If `k` is zero or negative, return an empty list.
    If `k >= len(items)` or `None`, return all items.
    """
    if k is None or k >= len(population):
        return list(population)
    if k <= 0:
        return []
    return rng.sample(population, k)


def mean(lst: Sequence[float]) -> float:
    """Calculate mean of `lst`. Returns 0 if `lst` is empty."""
    if not lst:
        return 0.0
    return sum(lst) / len(lst)
