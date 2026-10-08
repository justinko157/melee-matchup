"""Collect one week of start.gg data and publish it to Kafka."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from pipeline.ingest.envelope import build_dlq_envelope, build_envelope
from pipeline.ingest.queries import EVENT_SETS, TOURNAMENTS_BY_GAME
from pipeline.ingest.startgg import (
    MELEE_VIDEOGAME_ID,
    InvalidTokenError,
    StartGGError,
    operation_name,
)
from pipeline.ingest.windows import week_bounds
from pipeline.topics import DLQ_TOPIC, SETS_TOPIC, TOURNAMENTS_TOPIC

logger = logging.getLogger(__name__)

TOURNAMENTS_PER_PAGE = 50
SETS_PER_PAGE = 15


class Paginator(Protocol):
    def paginate(self, query: str, variables: dict, data_path: list[str]) -> list[dict]: ...


class Publisher(Protocol):
    def publish(self, topic: str, key: str, value: dict) -> None: ...


@dataclass
class WeekStats:
    tournaments: int = 0
    events: int = 0
    sets: int = 0
    dlq_messages: int = 0


def is_eligible_tournament(tournament: dict, min_attendees: int) -> bool:
    return (
        not tournament.get("isOnline", False)
        and (tournament.get("numAttendees") or 0) >= min_attendees
    )


def melee_events(tournament: dict) -> list[dict]:
    return [
        event
        for event in tournament.get("events") or []
        if (event.get("videogame") or {}).get("id") == MELEE_VIDEOGAME_ID
    ]


def collect_week(
    client: Paginator,
    publisher: Publisher,
    partition_week: str,
    *,
    run_id: str,
    min_attendees: int = 50,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> WeekStats:
    """Publish a week's eligible tournaments and their Melee sets.

    An event whose sets cannot be fetched is dead-lettered and the week
    continues. An invalid token or a failed discovery query raises.
    """
    after_date, before_date = week_bounds(partition_week)
    discovery_vars = {
        "videogameId": MELEE_VIDEOGAME_ID,
        "afterDate": after_date,
        "beforeDate": before_date,
        "perPage": TOURNAMENTS_PER_PAGE,
    }
    common = {"run_id": run_id, "partition_week": partition_week}
    stats = WeekStats()

    tournaments = client.paginate(TOURNAMENTS_BY_GAME, discovery_vars, ["tournaments"])
    for tournament in tournaments:
        if not is_eligible_tournament(tournament, min_attendees):
            continue
        if tournament.get("id") is None:
            logger.warning("Skipping tournament without an id: %s", tournament.get("name"))
            continue
        publisher.publish(
            TOURNAMENTS_TOPIC,
            str(tournament["id"]),
            build_envelope(
                entity="tournament",
                entity_id=tournament["id"],
                payload=tournament,
                query_name=operation_name(TOURNAMENTS_BY_GAME),
                query_variables=discovery_vars,
                ingested_at=now(),
                **common,
            ),
        )
        stats.tournaments += 1

        for event in melee_events(tournament):
            if event.get("id") is None:
                logger.warning("Skipping event without an id in tournament %s", tournament["id"])
                continue
            stats.events += 1
            set_vars = {"eventId": event["id"], "perPage": SETS_PER_PAGE}
            try:
                sets = client.paginate(EVENT_SETS, set_vars, ["event", "sets"])
            except InvalidTokenError:
                raise
            except StartGGError as exc:
                logger.error("Dead-lettering event %s: %s", event["id"], exc)
                publisher.publish(
                    DLQ_TOPIC,
                    str(event["id"]),
                    build_dlq_envelope(
                        entity="event",
                        entity_id=event["id"],
                        error=exc,
                        query_name=operation_name(EVENT_SETS),
                        query_variables=set_vars,
                        ingested_at=now(),
                        **common,
                    ),
                )
                stats.dlq_messages += 1
                continue

            for set_node in sets:
                if set_node.get("id") is None:
                    logger.warning("Skipping set without an id in event %s", event["id"])
                    continue
                publisher.publish(
                    SETS_TOPIC,
                    str(set_node["id"]),
                    build_envelope(
                        entity="set",
                        entity_id=set_node["id"],
                        payload=set_node,
                        query_name=operation_name(EVENT_SETS),
                        query_variables=set_vars,
                        ingested_at=now(),
                        **common,
                    ),
                )
                stats.sets += 1
    return stats
