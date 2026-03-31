"""start.gg GraphQL API client with rate limiting and retry logic."""

import logging
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_URL = "https://api.start.gg/gql/alpha"
MELEE_VIDEOGAME_ID = 1

# Rate limit: 80 requests per 60 seconds
MAX_REQUESTS_PER_WINDOW = 80
WINDOW_SECONDS = 60
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # exponential backoff: 2s, 4s, 8s


class StartGGClient:
    """GraphQL client for the start.gg API with built-in rate limiting."""

    def __init__(self, token: str | None = None):
        self.token = token or os.getenv("STARTGG_API_TOKEN")
        if not self.token:
            raise ValueError(
                "No API token provided. Set STARTGG_API_TOKEN in .env "
                "or pass token= to the client. "
                "Get a token at https://start.gg/admin/profile/developer"
            )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
        )
        self._request_timestamps: list[float] = []

    def _wait_for_rate_limit(self):
        """Block until we're under the rate limit."""
        now = time.time()
        # Remove timestamps older than the window
        self._request_timestamps = [
            t for t in self._request_timestamps if now - t < WINDOW_SECONDS
        ]
        if len(self._request_timestamps) >= MAX_REQUESTS_PER_WINDOW:
            oldest = self._request_timestamps[0]
            wait_time = WINDOW_SECONDS - (now - oldest) + 0.1
            logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
            time.sleep(wait_time)

    def query(self, query: str, variables: dict | None = None) -> dict:
        """Execute a GraphQL query with rate limiting and retries.

        Returns the 'data' portion of the response.
        Raises on HTTP errors or GraphQL errors after retries are exhausted.
        """
        self._wait_for_rate_limit()

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        for attempt in range(MAX_RETRIES):
            try:
                self._request_timestamps.append(time.time())
                resp = self.session.post(API_URL, json=payload, timeout=30)

                if resp.status_code == 429:
                    wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(f"429 Too Many Requests, retrying in {wait}s")
                    time.sleep(wait)
                    continue

                if resp.status_code >= 400:
                    logger.error(f"HTTP {resp.status_code}: {resp.text}")
                resp.raise_for_status()
                data = resp.json()

                if "errors" in data:
                    errors = data["errors"]
                    logger.error(f"GraphQL errors: {errors}")
                    raise RuntimeError(f"GraphQL errors: {errors}")

                return data["data"]

            except requests.exceptions.Timeout:
                wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"Request timed out, retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            except requests.exceptions.ConnectionError:
                wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"Connection error, retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)

        raise RuntimeError(f"Failed after {MAX_RETRIES} retries")

    def paginate(self, query: str, variables: dict, data_path: list[str]) -> list[dict]:
        """Auto-paginate a query that returns nodes with pageInfo.

        Automatically reduces page size on complexity errors (1000-object limit).

        Args:
            query: GraphQL query string (must accept $page and $perPage variables)
            variables: Base variables (page/perPage will be managed automatically)
            data_path: Path to the paginated field in the response,
                       e.g. ["tournaments"] or ["event", "sets"]

        Returns:
            All nodes concatenated across pages.
        """
        all_nodes = []
        page = 1
        per_page = variables.get("perPage", 50)

        while True:
            variables = {**variables, "page": page, "perPage": per_page}
            try:
                result = self.query(query, variables)
            except RuntimeError as e:
                if "complexity" in str(e).lower() and per_page > 5:
                    per_page = max(5, per_page // 2)
                    logger.warning(f"Query too complex, reducing page size to {per_page}")
                    continue
                raise

            # Navigate to the paginated field
            obj = result
            for key in data_path:
                if obj is None:
                    return all_nodes
                obj = obj.get(key)

            if obj is None:
                break

            nodes = obj.get("nodes", [])
            page_info = obj.get("pageInfo", {})

            all_nodes.extend(nodes)

            total_pages = page_info.get("totalPages", 1)
            logger.info(f"Page {page}/{total_pages} — {len(all_nodes)} items so far")

            if page >= total_pages:
                break
            page += 1

        return all_nodes
