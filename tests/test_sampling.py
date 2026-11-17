"""Unit tests for sampling helpers in `genret.utils`."""

import random

from genret.utils import sample


class TestSample:
    """Tests for `sample`."""

    def test_samples_when_count_smaller(self) -> None:
        """Should sample when `k < len(population)`."""
        population = list(range(100))
        rng = random.Random(42)

        result = sample(rng, population, 10)

        assert len(result) == 10
        assert all(item in population for item in result)

    def test_returns_original_when_count_larger(self) -> None:
        """Should return the original items when `k >= len(population)`."""
        population = [1, 2, 3]
        rng = random.Random(42)

        result = sample(rng, population, 10)

        assert result == population

    def test_returns_original_when_count_none(self) -> None:
        """Should return the original items when `k` is `None`."""
        population = [1, 2, 3, 4, 5]
        rng = random.Random(42)

        result = sample(rng, population, None)

        assert result == population

    def test_returns_empty_when_count_non_positive(self) -> None:
        """Should return an empty list when `k` is zero or negative."""
        population = [1, 2, 3]
        rng = random.Random(42)

        assert sample(rng, population, 0) == []
        assert sample(rng, population, -1) == []

    def test_reproducible_with_seed(self) -> None:
        """Should produce the same results with the same seed."""
        population = list(range(100))

        result1 = sample(random.Random(42), population, 10)
        result2 = sample(random.Random(42), population, 10)

        assert result1 == result2
