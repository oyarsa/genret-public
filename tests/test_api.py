"""Unit tests for api.py."""

import pytest

from genret.s2.api import extract_pdf_url
from genret.s2.types import S2ExternalIds, S2OpenAccessPdf, S2Paper


class TestExtractPdfUrl:
    """Tests for extract_pdf_url function."""

    @pytest.mark.parametrize(
        ("paper", "expected_url", "expected_source"),
        [
            # S2 openAccessPdf
            (
                S2Paper(
                    open_access_pdf=S2OpenAccessPdf(url="https://example.com/paper.pdf")
                ),
                "https://example.com/paper.pdf",
                "S2",
            ),
            # ArXiv fallback
            (
                S2Paper(external_ids=S2ExternalIds(arxiv="2401.12345")),
                "https://arxiv.org/pdf/2401.12345.pdf",
                "ArXiv",
            ),
            # ACL Anthology fallback
            (
                S2Paper(
                    external_ids=S2ExternalIds(doi="10.18653/v1/2024.acl-long.123")
                ),
                "https://aclanthology.org/2024.acl-long.123.pdf",
                "ACL",
            ),
            # ACL Anthology - EMNLP format
            (
                S2Paper(
                    external_ids=S2ExternalIds(doi="10.18653/v1/2023.emnlp-main.456"),
                ),
                "https://aclanthology.org/2023.emnlp-main.456.pdf",
                "ACL",
            ),
            # ACL Anthology - old format
            (
                S2Paper(external_ids=S2ExternalIds(doi="10.18653/v1/P19-1001")),
                "https://aclanthology.org/P19-1001.pdf",
                "ACL",
            ),
            # S2 preferred over ArXiv
            (
                S2Paper(
                    open_access_pdf=S2OpenAccessPdf(url="https://s2.com/paper.pdf"),
                    external_ids=S2ExternalIds(arxiv="2401.12345"),
                ),
                "https://s2.com/paper.pdf",
                "S2",
            ),
            # ArXiv preferred over ACL
            (
                S2Paper(
                    external_ids=S2ExternalIds(
                        arxiv="2401.12345",
                        doi="10.18653/v1/2024.acl-long.123",
                    ),
                ),
                "https://arxiv.org/pdf/2401.12345.pdf",
                "ArXiv",
            ),
            # Empty S2 URL falls back to ArXiv
            (
                S2Paper(
                    open_access_pdf=S2OpenAccessPdf(url=""),
                    external_ids=S2ExternalIds(arxiv="2401.12345"),
                ),
                "https://arxiv.org/pdf/2401.12345.pdf",
                "ArXiv",
            ),
            # Non-ACL DOI ignored
            (
                S2Paper(external_ids=S2ExternalIds(doi="10.1234/other.paper")),
                None,
                None,
            ),
            # Missing externalIds
            (
                S2Paper(),
                None,
                None,
            ),
            # None externalIds
            (
                S2Paper(external_ids=None),
                None,
                None,
            ),
            # Empty openAccessPdf (url defaults to "")
            (
                S2Paper(open_access_pdf=S2OpenAccessPdf()),
                None,
                None,
            ),
        ],
        ids=[
            "s2_pdf",
            "arxiv_fallback",
            "acl_fallback",
            "acl_emnlp_format",
            "acl_old_format",
            "s2_over_arxiv",
            "arxiv_over_acl",
            "empty_url_fallback",
            "non_acl_doi_ignored",
            "missing_external_ids",
            "none_external_ids",
            "empty_open_access_pdf",
        ],
    )
    def test_extract_pdf_url(
        self,
        paper: S2Paper,
        expected_url: str | None,
        expected_source: str | None,
    ) -> None:
        """Should extract PDF URL with appropriate fallbacks."""
        url, source = extract_pdf_url(paper)
        assert url == expected_url
        assert source == expected_source
