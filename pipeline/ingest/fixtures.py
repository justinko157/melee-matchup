"""Record and replay start.gg pagination results, for tests and the smoke test."""

import json
from pathlib import Path

from pipeline.ingest.envelope import strip_pagination
from pipeline.ingest.startgg import operation_name


def fixture_key(query: str, variables: dict) -> str:
    """Identify a paginated query by operation name and non-paging variables."""
    return f"{operation_name(query)}:{json.dumps(strip_pagination(variables), sort_keys=True)}"


class RecordingClient:
    """Wraps a real client and records every paginate() result."""

    def __init__(self, inner):
        self._inner = inner
        self.recorded: dict[str, list] = {}

    @property
    def api_calls(self) -> int:
        return self._inner.api_calls

    def paginate(self, query: str, variables: dict, data_path: list[str]) -> list[dict]:
        nodes = self._inner.paginate(query, variables, data_path)
        self.recorded[fixture_key(query, variables)] = nodes
        return nodes

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.recorded, separators=(",", ":"), sort_keys=True), encoding="utf-8"
        )


class FixtureClient:
    """Replays a recorded week without calling the API."""

    api_calls = 0

    def __init__(self, path: str | Path):
        self._data = json.loads(Path(path).read_text(encoding="utf-8"))

    def paginate(self, query: str, variables: dict, data_path: list[str]) -> list[dict]:
        key = fixture_key(query, variables)
        if key not in self._data:
            raise KeyError(f"No recorded response for {key}")
        return self._data[key]
