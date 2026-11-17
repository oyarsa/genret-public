"""Typed wrapper for rank_bm25 (no type stubs)."""

# pyright: basic

from rank_bm25 import BM25Okapi


class BM25Index:
    """Typed facade over ``rank_bm25.BM25Okapi``."""

    def __init__(self, tokenized_docs: list[list[str]]) -> None:
        """Build a BM25 index from pre-tokenised documents.

        Args:
            tokenized_docs: List of tokenised documents (each a list of tokens).
        """
        self._index: BM25Okapi = BM25Okapi(tokenized_docs)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        """Score all documents against a query.

        Args:
            query_tokens: Tokenised query.

        Returns:
            List of BM25 scores, one per document in index order.
        """
        scores = self._index.get_scores(query_tokens)
        return [float(s) for s in scores]
