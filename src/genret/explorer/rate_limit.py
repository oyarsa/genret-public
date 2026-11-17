"""Per-client rate limiting for the endpoints that spend API credit.

Graft is served without authentication, and `/api/explain` and
`/api/recommend` both call an LLM when their result is not cached. A
sliding window per client IP bounds what a single visitor (or a scanner
that found the host) can spend, and a global window bounds the damage
from many addresses at once.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

GLOBAL_KEY = "*"


@dataclass(frozen=True, kw_only=True)
class RateLimitConfig:
    """Limits for the paid endpoints, per sliding window."""

    per_client: int = 40
    """Requests allowed per client per window. 0 disables limiting."""
    total: int = 400
    """Requests allowed across all clients per window."""
    window_seconds: float = 3600.0


class RateLimiter:
    """Sliding-window request counter keyed by client.

    In-process and therefore per-container, which is enough for a
    single-instance deployment; a second replica would need shared state.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)

    @property
    def enabled(self) -> bool:
        """Whether limiting is switched on."""
        return self._config.per_client > 0

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits[key]
        cutoff = now - self._config.window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()
        return hits

    def check(self, client: str, now: float | None = None) -> float | None:
        """Record a request, or report how long the caller must wait.

        Args:
            client: Identifier for the caller, usually its IP address.
            now: Current time; defaults to the monotonic clock.

        Returns:
            None when the request is allowed, otherwise the number of
            seconds until the oldest counted request falls out of the
            window.
        """
        if not self.enabled:
            return None

        now = time.monotonic() if now is None else now
        for key, limit in (
            (client, self._config.per_client),
            (GLOBAL_KEY, self._config.total),
        ):
            hits = self._prune(key, now)
            if len(hits) >= limit:
                retry_after = hits[0] + self._config.window_seconds - now
                logger.warning(
                    "Rate limit reached for %s (%d in window)", key, len(hits)
                )
                return max(retry_after, 1.0)

        for key in (client, GLOBAL_KEY):
            self._hits[key].append(now)
        return None
