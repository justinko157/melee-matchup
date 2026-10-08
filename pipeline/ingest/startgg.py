"""start.gg GraphQL client: rate limiting, retries, error classification, pagination."""

import logging
import re
import time
from collections import deque
from collections.abc import Callable

import requests

logger = logging.getLogger(__name__)

API_URL = "https://api.start.gg/gql/alpha"
MELEE_VIDEOGAME_ID = 1

# start.gg allows 80 requests per 60 seconds
MAX_REQUESTS_PER_WINDOW = 80
WINDOW_SECONDS = 60.0
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2.0  # waits 2s, 4s, 8s between attempts
MIN_PAGE_SIZE = 5
TOKEN_HELP = "Create a new token at https://start.gg/admin/profile/developer"


class StartGGError(Exception):
    """Base class for start.gg failures."""


class InvalidTokenError(StartGGError):
    """The token is missing or rejected. Never retried; fails the run."""


class RetriesExhaustedError(StartGGError):
    """Transient failures (429, 5xx, timeouts) persisted past MAX_ATTEMPTS."""


class RequestRejectedError(StartGGError):
    """A non-retryable HTTP 4xx other than an auth failure."""


class GraphQLError(StartGGError):
    """The response contained GraphQL errors."""


class ComplexityError(GraphQLError):
    """The query asked for too many objects per page."""


class PaginationCapError(GraphQLError):
    """start.gg refuses to page past its 10,000th result."""


def operation_name(query: str) -> str:
    """Return the GraphQL operation name, e.g. 'EventSets'."""
    match = re.search(r"\bquery\s+(\w+)", query)
    if not match:
        raise ValueError("GraphQL query has no operation name")
    return match.group(1)


def _classify_graphql_errors(errors: list) -> GraphQLError:
    text = str(errors)
    if "complexity" in text.lower():
        return ComplexityError(text)
    if "10,000" in text or "10000" in text:
        return PaginationCapError(text)
    return GraphQLError(text)


class RateLimiter:
    """Sliding-window limiter: at most max_requests per window_seconds."""

    def __init__(
        self,
        max_requests: int = MAX_REQUESTS_PER_WINDOW,
        window_seconds: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._max = max_requests
        self._window = window_seconds
        self._clock = clock
        self._sleep = sleep
        self._timestamps: deque[float] = deque()

    def _evict(self, now: float) -> None:
        while self._timestamps and now - self._timestamps[0] >= self._window:
            self._timestamps.popleft()

    def acquire(self) -> None:
        now = self._clock()
        self._evict(now)
        if len(self._timestamps) >= self._max:
            wait = self._window - (now - self._timestamps[0])
            logger.info("Rate limit reached, waiting %.1fs", wait)
            self._sleep(wait)
            now = self._clock()
            self._evict(now)
        self._timestamps.append(now)


class StartGGClient:
    """GraphQL client for the start.gg API."""

    def __init__(
        self,
        token: str,
        session: requests.Session | None = None,
        rate_limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not token:
            raise InvalidTokenError(f"STARTGG_API_TOKEN is empty. Set it in .env. {TOKEN_HELP}")
        self._session = session or requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )
        self._rate_limiter = rate_limiter or RateLimiter()
        self._sleep = sleep
        self.api_calls = 0

    def query(self, query: str, variables: dict) -> dict:
        """Run one GraphQL request and return its 'data'."""
        payload = {"query": query, "variables": variables}
        last_error: object = None
        for attempt in range(MAX_ATTEMPTS):
            if attempt:
                wait = BACKOFF_BASE_SECONDS**attempt
                logger.warning(
                    "Retrying %s in %.0fs after: %s", operation_name(query), wait, last_error
                )
                self._sleep(wait)
            self._rate_limiter.acquire()
            self.api_calls += 1
            try:
                resp = self._session.post(API_URL, json=payload, timeout=30)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = repr(exc)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                continue
            if resp.status_code == 401 or (
                resp.status_code == 400 and "authentication token" in resp.text.lower()
            ):
                raise InvalidTokenError(
                    f"start.gg rejected STARTGG_API_TOKEN (HTTP {resp.status_code}). {TOKEN_HELP}"
                )
            if resp.status_code >= 400:
                raise RequestRejectedError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            body = resp.json()
            if body.get("errors"):
                raise _classify_graphql_errors(body["errors"])
            return body["data"]
        raise RetriesExhaustedError(
            f"{operation_name(query)} failed after {MAX_ATTEMPTS} attempts: {last_error}"
        )

    def paginate(self, query: str, variables: dict, data_path: list[str]) -> list[dict]:
        """Fetch every page of a paginated field and return all nodes.

        On a complexity error the page size is halved and pagination restarts
        at page 1, because page boundaries move when the page size changes.
        """
        nodes: list[dict] = []
        page = 1
        per_page = variables.get("perPage", 50)
        while True:
            try:
                data = self.query(query, {**variables, "page": page, "perPage": per_page})
            except ComplexityError:
                if per_page <= MIN_PAGE_SIZE:
                    raise
                per_page = max(MIN_PAGE_SIZE, per_page // 2)
                logger.warning("Query too complex, restarting with page size %d", per_page)
                nodes, page = [], 1
                continue
            obj = data
            for key in data_path:
                obj = (obj or {}).get(key)
            if obj is None:
                return nodes
            nodes.extend(obj.get("nodes") or [])
            total_pages = (obj.get("pageInfo") or {}).get("totalPages") or 1
            if page >= total_pages:
                return nodes
            page += 1
