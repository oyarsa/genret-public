"""Tests for venue normalisation logic."""

from genret.config import normalise_venue


class TestFindingsResolution:
    """Tests for resolving bare 'Findings' venues via ACL IDs."""

    def test_findings_with_emnlp_acl_id(self) -> None:
        """Findings + EMNLP ACL ID resolves to 'Findings of EMNLP'."""
        assert (
            normalise_venue("Findings", acl_id="2020.findings-emnlp.82")
            == "Findings of EMNLP"
        )

    def test_findings_with_acl_acl_id(self) -> None:
        """Findings + ACL ACL ID resolves to 'Findings of ACL'."""
        assert (
            normalise_venue("Findings", acl_id="2021.findings-acl.100")
            == "Findings of ACL"
        )

    def test_findings_with_naacl_acl_id(self) -> None:
        """Findings + NAACL ACL ID resolves to 'Findings of NAACL'."""
        assert (
            normalise_venue("Findings", acl_id="2022.findings-naacl.55")
            == "Findings of NAACL"
        )

    def test_findings_with_eacl_acl_id(self) -> None:
        """Findings + EACL ACL ID resolves to 'Findings of EACL'."""
        assert (
            normalise_venue("Findings", acl_id="2023.findings-eacl.12")
            == "Findings of EACL"
        )

    def test_findings_with_aacl_acl_id(self) -> None:
        """Findings + AACL ACL ID resolves to 'Findings of AACL-IJCNLP'."""
        assert (
            normalise_venue("Findings", acl_id="2022.findings-aacl.33")
            == "Findings of AACL-IJCNLP"
        )

    def test_findings_without_acl_id(self) -> None:
        """Findings without ACL ID falls back to 'Findings (unknown)'."""
        assert normalise_venue("Findings") == "Findings (unknown)"

    def test_findings_with_none_acl_id(self) -> None:
        """Findings with explicit None ACL ID falls back to unknown."""
        assert normalise_venue("Findings", acl_id=None) == "Findings (unknown)"

    def test_findings_with_unrecognised_acl_id(self) -> None:
        """Findings with unrecognised conference fragment falls back to unknown."""
        assert (
            normalise_venue("Findings", acl_id="2024.findings-zzzz.1")
            == "Findings (unknown)"
        )


class TestAliasMapping:
    """Tests for mapping known venue aliases to canonical names."""

    def test_naacl_hlt_maps_to_naacl(self) -> None:
        """NAACL-HLT is normalised to NAACL."""
        assert normalise_venue("NAACL-HLT") == "NAACL"

    def test_long_emnlp_name(self) -> None:
        """Long EMNLP venue name maps to canonical short name."""
        assert (
            normalise_venue(
                "Conference on Empirical Methods in Natural Language Processing"
            )
            == "EMNLP"
        )

    def test_long_acl_name(self) -> None:
        """Long ACL venue name maps to canonical short name."""
        assert (
            normalise_venue(
                "Annual Meeting of the Association for Computational Linguistics"
            )
            == "ACL"
        )

    def test_ijcnlp_maps_to_aacl(self) -> None:
        """IJCNLP is normalised to AACL-IJCNLP."""
        assert normalise_venue("IJCNLP") == "AACL-IJCNLP"

    def test_sigdial_conferences_maps_to_sigdial(self) -> None:
        """'SIGDIAL Conferences' is normalised to SIGDIAL."""
        assert normalise_venue("SIGDIAL Conferences") == "SIGDIAL"

    def test_canonical_name_unchanged(self) -> None:
        """A canonical short name that is also a venue pattern stays the same."""
        assert normalise_venue("EMNLP") == "EMNLP"
        assert normalise_venue("ACL") == "ACL"
        assert normalise_venue("NAACL") == "NAACL"


class TestPassthrough:
    """Tests for unknown venues passing through unchanged."""

    def test_unknown_venue_passes_through(self) -> None:
        """An unrecognised venue is returned as-is."""
        assert normalise_venue("Some Workshop 2024") == "Some Workshop 2024"

    def test_empty_string_passes_through(self) -> None:
        """An empty string is returned as-is."""
        assert normalise_venue("") == ""

    def test_non_findings_with_acl_id_uses_alias(self) -> None:
        """A non-Findings venue ignores the ACL ID and uses alias mapping."""
        assert normalise_venue("NAACL-HLT", acl_id="2020.naacl-main.100") == "NAACL"
