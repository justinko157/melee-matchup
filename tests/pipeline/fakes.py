"""Test doubles for the start.gg client and Kafka publisher."""

from pipeline.ingest.startgg import operation_name


class FakePaginator:
    """Stands in for StartGGClient.paginate.

    sets_by_event maps an event ID to its set nodes, or to an exception that
    paginating that event should raise.
    """

    def __init__(self, tournaments, sets_by_event=None, discovery_error=None):
        self.tournaments = tournaments
        self.sets_by_event = sets_by_event or {}
        self.discovery_error = discovery_error
        self.calls = []
        self.api_calls = 0

    def paginate(self, query, variables, data_path):
        name = operation_name(query)
        self.calls.append((name, dict(variables)))
        self.api_calls += 1
        if name == "TournamentsByVideogame":
            if self.discovery_error:
                raise self.discovery_error
            return self.tournaments
        result = self.sets_by_event.get(variables["eventId"], [])
        if isinstance(result, Exception):
            raise result
        return result


class FakePublisher:
    def __init__(self):
        self.messages = []
        self.flushed = False

    def publish(self, topic, key, value):
        self.messages.append((topic, key, value))

    def flush(self, timeout=60.0):
        self.flushed = True

    def on(self, topic):
        return [m for m in self.messages if m[0] == topic]


def tournament(tid, *, attendees=100, online=False, events=None):
    return {
        "id": tid,
        "name": f"T{tid}",
        "numAttendees": attendees,
        "isOnline": online,
        "events": events,
    }


def melee_event(eid):
    return {"id": eid, "name": f"E{eid}", "videogame": {"id": 1}}
