"""Typed wrapper for aiofiles protocol conformance."""

from typing import Any, Protocol, cast


class AsyncWritable(Protocol):
    """Protocol for files that can async write and flush."""

    async def write(self, data: str, /) -> int: ...
    async def flush(self) -> None: ...


def as_async_writable(file: Any) -> AsyncWritable:
    """Cast an aiofiles handle to the ``AsyncWritable`` protocol.

    ``AsyncTextIOWrapper`` satisfies the protocol at runtime but stubs
    disagree on parameter names. This wrapper hides that.

    Args:
        file: Async file handle from ``aiofiles.open``.

    Returns:
        The same handle, typed as ``AsyncWritable``.
    """
    return cast(AsyncWritable, file)
