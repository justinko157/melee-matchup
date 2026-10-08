import pytest
import requests

from pipeline.ingest.startgg import (
    MAX_ATTEMPTS,
    ComplexityError,
    GraphQLError,
    InvalidTokenError,
    PaginationCapError,
    RateLimiter,
    RequestRejectedError,
    RetriesExhaustedError,
    StartGGClient,
    operation_name,
)

TOKEN = "secret-token-value"
QUERY = "query EventSets($eventId: ID!) { event(id: $eventId) { id } }"
PAGED_QUERY = "query EventSets($eventId: ID!, $page: Int!, $perPage: Int!) { x }"


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        return self._body


class FakeSession:
    """Returns queued responses (or raises queued exceptions) in order."""

    def __init__(self, *responses):
        self.headers = {}
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append(json)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ok(data):
    return FakeResponse(200, {"data": data})


def make_client(*responses):
    session = FakeSession(*responses)
    sleeps = []
    client = StartGGClient(
        TOKEN,
        session=session,
        rate_limiter=RateLimiter(max_requests=10_000),
        sleep=sleeps.append,
    )
    return client, session, sleeps


def test_operation_name():
    assert operation_name(QUERY) == "EventSets"


def test_empty_token_raises_invalid_token():
    with pytest.raises(InvalidTokenError, match="STARTGG_API_TOKEN"):
        StartGGClient("")


def test_query_returns_data_and_sets_auth_header():
    client, session, _ = make_client(ok({"event": {"id": 1}}))
    assert client.query(QUERY, {"eventId": 1}) == {"event": {"id": 1}}
    assert session.headers["Authorization"] == f"Bearer {TOKEN}"
    assert client.api_calls == 1


def test_invalid_token_400_fails_fast_without_retry():
    client, session, _ = make_client(
        FakeResponse(400, text='{"message":"Invalid authentication token"}')
    )
    with pytest.raises(InvalidTokenError):
        client.query(QUERY, {})
    assert len(session.calls) == 1


def test_401_fails_fast():
    client, session, _ = make_client(FakeResponse(401, text="Unauthorized"))
    with pytest.raises(InvalidTokenError):
        client.query(QUERY, {})
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "first",
    [
        FakeResponse(429),
        FakeResponse(502),
        requests.exceptions.Timeout(),
        requests.exceptions.ConnectionError(),
    ],
)
def test_transient_failures_are_retried(first):
    client, session, sleeps = make_client(first, ok({"event": None}))
    assert client.query(QUERY, {}) == {"event": None}
    assert len(session.calls) == 2
    assert sleeps == [2.0]


def test_gives_up_after_max_attempts():
    client, session, sleeps = make_client(*[FakeResponse(503)] * MAX_ATTEMPTS)
    with pytest.raises(RetriesExhaustedError, match="EventSets"):
        client.query(QUERY, {})
    assert len(session.calls) == MAX_ATTEMPTS
    assert sleeps == [2.0, 4.0, 8.0]


def test_other_4xx_is_not_retried():
    client, session, _ = make_client(FakeResponse(404, text="not found"))
    with pytest.raises(RequestRejectedError, match="404"):
        client.query(QUERY, {})
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    ("message", "error_class"),
    [
        ("Your query complexity is too high", ComplexityError),
        ("Cannot query more than 10,000th entry", PaginationCapError),
        ("Something else broke", GraphQLError),
    ],
)
def test_graphql_errors_are_classified(message, error_class):
    client, _, _ = make_client(FakeResponse(200, {"errors": [{"message": message}]}))
    with pytest.raises(error_class):
        client.query(QUERY, {})


def test_error_messages_never_contain_the_token():
    client, _, _ = make_client(FakeResponse(401, text="Unauthorized"))
    with pytest.raises(InvalidTokenError) as invalid:
        client.query(QUERY, {})
    client, _, _ = make_client(*[FakeResponse(500)] * MAX_ATTEMPTS)
    with pytest.raises(RetriesExhaustedError) as exhausted:
        client.query(QUERY, {})
    assert TOKEN not in str(invalid.value)
    assert TOKEN not in str(exhausted.value)


def test_rate_limiter_sleeps_until_window_frees():
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(max_requests=2, window_seconds=10.0, clock=lambda: now[0], sleep=sleep)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
    assert sleeps == [10.0]


def page(nodes, page_num, total_pages):
    return ok(
        {
            "event": {
                "sets": {"nodes": nodes, "pageInfo": {"totalPages": total_pages, "page": page_num}}
            }
        }
    )


def test_paginate_concatenates_pages():
    client, session, _ = make_client(page([{"id": 1}], 1, 2), page([{"id": 2}], 2, 2))
    nodes = client.paginate(PAGED_QUERY, {"eventId": 5, "perPage": 15}, ["event", "sets"])
    assert nodes == [{"id": 1}, {"id": 2}]
    assert [c["variables"]["page"] for c in session.calls] == [1, 2]


def test_paginate_halves_page_size_and_restarts_on_complexity():
    complexity = FakeResponse(200, {"errors": [{"message": "query complexity is too high"}]})
    client, session, _ = make_client(page([{"id": 1}], 1, 2), complexity, page([{"id": 1}], 1, 1))
    nodes = client.paginate(PAGED_QUERY, {"eventId": 5, "perPage": 20}, ["event", "sets"])
    assert nodes == [{"id": 1}]
    assert [(c["variables"]["page"], c["variables"]["perPage"]) for c in session.calls] == [
        (1, 20),
        (2, 20),
        (1, 10),
    ]


def test_paginate_raises_on_pagination_cap():
    cap = FakeResponse(200, {"errors": [{"message": "Cannot query more than 10,000th entry"}]})
    client, _, _ = make_client(page([{"id": 1}], 1, 3), cap)
    with pytest.raises(PaginationCapError):
        client.paginate(PAGED_QUERY, {"eventId": 5, "perPage": 15}, ["event", "sets"])


def test_paginate_null_nodes_is_empty():
    client, _, _ = make_client(page(None, 1, 1))
    assert client.paginate(PAGED_QUERY, {"eventId": 5}, ["event", "sets"]) == []


def test_paginate_missing_path_is_empty():
    client, _, _ = make_client(ok({"event": None}))
    assert client.paginate(PAGED_QUERY, {"eventId": 5}, ["event", "sets"]) == []
