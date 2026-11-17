"""Pydantic models for Semantic Scholar API responses.

These models cover the paper search, references, and citations endpoints.
Field names use snake_case; validation aliases match the camelCase API JSON keys.
"""

from pydantic import BaseModel, ConfigDict, Field


class S2OpenAccessPdf(BaseModel, frozen=True):
    """Open access PDF information from a Semantic Scholar paper."""

    url: str = ""


class S2ExternalIds(BaseModel, frozen=True):
    """External identifiers from a Semantic Scholar paper."""

    model_config = ConfigDict(populate_by_name=True)

    arxiv: str | None = Field(default=None, validation_alias="ArXiv")
    doi: str | None = Field(default=None, validation_alias="DOI")
    acl: str | None = Field(default=None, validation_alias="ACL")


class S2Author(BaseModel, frozen=True):
    """Author information from a Semantic Scholar paper."""

    model_config = ConfigDict(populate_by_name=True)

    author_id: str | None = Field(default=None, validation_alias="authorId")
    name: str = ""


class S2Paper(BaseModel, frozen=True):
    """Paper data from the Semantic Scholar search API.

    All fields are optional with defaults so the model can handle partial
    responses (different field selections return different subsets).
    """

    model_config = ConfigDict(populate_by_name=True)

    paper_id: str = Field(default="", validation_alias="paperId")
    corpus_id: int | None = Field(default=None, validation_alias="corpusId")
    title: str = ""
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = Field(default=None, validation_alias="citationCount")
    fields_of_study: list[str] | None = Field(
        default=None, validation_alias="fieldsOfStudy"
    )
    url: str = ""
    open_access_pdf: S2OpenAccessPdf | None = Field(
        default=None, validation_alias="openAccessPdf"
    )
    external_ids: S2ExternalIds | None = Field(
        default=None, validation_alias="externalIds"
    )
    # Extended fields (only present with certain field selections)
    abstract: str | None = None
    authors: list[S2Author] | None = None


class S2BulkSearchResponse(BaseModel, frozen=True):
    """Response from the Semantic Scholar bulk search endpoint."""

    data: list[S2Paper] = []
    token: str | None = None
    total: int | None = None


class S2SearchResponse(BaseModel, frozen=True):
    """Response from the Semantic Scholar relevance search endpoint."""

    total: int = 0
    offset: int = 0
    next: int | None = None
    data: list[S2Paper] = []


class S2PaperRef(BaseModel, frozen=True):
    """Minimal paper reference with just the paper ID."""

    model_config = ConfigDict(populate_by_name=True)

    paper_id: str | None = Field(default=None, validation_alias="paperId")
    corpus_id: int | None = Field(default=None, validation_alias="corpusId")


class S2RelatedPaperItem(BaseModel, frozen=True):
    """Item from the references or citations endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    cited_paper: S2PaperRef | None = Field(default=None, validation_alias="citedPaper")
    citing_paper: S2PaperRef | None = Field(
        default=None, validation_alias="citingPaper"
    )


class S2RelatedPapersResponse(BaseModel, frozen=True):
    """Response from references or citations endpoint."""

    data: list[S2RelatedPaperItem] = []
