"""Typed wrapper for pymupdf4llm (no type stubs)."""

# pyright: basic

import pymupdf4llm  # type: ignore[import-untyped]


def pdf_to_markdown(path: str) -> str:
    """Extract markdown from a PDF file.

    Args:
        path: Filesystem path to the PDF.

    Returns:
        Markdown string extracted from the PDF.
    """
    result = pymupdf4llm.to_markdown(path)  # type: ignore[no-any-return]
    assert isinstance(result, str)
    return result
