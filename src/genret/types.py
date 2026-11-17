"""Shared domain-specific NewType definitions for the genret package."""

from typing import NewType

from genret._shed.beartype import is_type as is_type

CorpusId = NewType("CorpusId", int)
"""Semantic Scholar corpus ID (integer, unique per paper in the S2 corpus)."""

PaperId = NewType("PaperId", str)
"""Semantic Scholar paper ID (opaque string)."""

AuthorId = NewType("AuthorId", str)
"""Semantic Scholar author ID (opaque string)."""

Year = NewType("Year", int)
"""Publication year of a paper."""
