"""Data collection pipeline: start.gg API → SQLite database."""

import logging

from tqdm import tqdm

from .api_client import MELEE_VIDEOGAME_ID, StartGGClient
from .database import MeleeDB
from .queries import EVENT_SETS, TOURNAMENTS_BY_GAME

logger = logging.getLogger(__name__)


def _parse_slot(slot: dict) -> dict:
    """Extract player info from a set slot."""
    entrant = slot.get("entrant") or {}
    seed_obj = slot.get("seed") or {}
    standing = slot.get("standing") or {}
    stats = standing.get("stats") or {}
    score_obj = stats.get("score") or {}

    # Get the stable player ID and tag from participants
    player_id = None
    gamer_tag = None
    participants = entrant.get("participants") or []
    if participants:
        player = participants[0].get("player") or {}
        player_id = player.get("id")
        gamer_tag = player.get("gamerTag")

    return {
        "entrant_id": entrant.get("id"),
        "player_id": player_id,
        "gamer_tag": gamer_tag,
        "seed": seed_obj.get("seedNum") or entrant.get("initialSeedNum"),
        "score": score_obj.get("value"),
        "placement": standing.get("placement"),
    }


def _parse_set(set_node: dict, event_id: int) -> tuple[dict | None, list[dict]]:
    """Parse a set node into a set record and game records.

    Returns (set_dict, games_list). Returns (None, []) if the set is
    invalid (e.g., a bye or DQ with no real data).
    """
    slots = set_node.get("slots") or []
    if len(slots) < 2:
        return None, []

    p1 = _parse_slot(slots[0])
    p2 = _parse_slot(slots[1])

    # Skip sets where we can't identify both players
    if not p1["player_id"] or not p2["player_id"]:
        return None, []

    # Skip non-completed sets
    state = set_node.get("state")
    if state != 3:
        return None, []

    # Determine winner by matching winnerId to entrant IDs
    winner_id = set_node.get("winnerId")
    winner_player_id = None
    if winner_id == p1["entrant_id"]:
        winner_player_id = p1["player_id"]
    elif winner_id == p2["entrant_id"]:
        winner_player_id = p2["player_id"]

    phase_group = set_node.get("phaseGroup") or {}
    phase = phase_group.get("phase") or {}

    set_record = {
        "id": set_node["id"],
        "event_id": event_id,
        "round": set_node.get("round"),
        "full_round_text": set_node.get("fullRoundText"),
        "phase_name": phase.get("name"),
        "completed_at": set_node.get("completedAt"),
        "total_games": set_node.get("totalGames"),
        "display_score": set_node.get("displayScore"),
        "entrant1_id": p1["entrant_id"],
        "entrant2_id": p2["entrant_id"],
        "player1_id": p1["player_id"],
        "player2_id": p2["player_id"],
        "winner_player_id": winner_player_id,
        "player1_score": p1["score"],
        "player2_score": p2["score"],
        "player1_seed": p1["seed"],
        "player2_seed": p2["seed"],
        "state": state,
    }

    # Parse individual games
    games = []
    for game_node in set_node.get("games") or []:
        game = _parse_game(game_node, set_node["id"], p1, p2)
        if game:
            games.append(game)

    return set_record, games


def _parse_game(
    game_node: dict, set_id: int, p1: dict, p2: dict
) -> dict | None:
    """Parse a game node into a game record."""
    if not game_node.get("id"):
        return None

    # Match character selections to players via entrant IDs
    p1_char_id, p1_char_name = None, None
    p2_char_id, p2_char_name = None, None

    for sel in game_node.get("selections") or []:
        sel_entrant = sel.get("entrant") or {}
        character = sel.get("character") or {}
        eid = sel_entrant.get("id")
        if eid == p1["entrant_id"]:
            p1_char_id = character.get("id")
            p1_char_name = character.get("name")
        elif eid == p2["entrant_id"]:
            p2_char_id = character.get("id")
            p2_char_name = character.get("name")

    stage = game_node.get("stage") or {}

    return {
        "id": game_node["id"],
        "set_id": set_id,
        "game_number": game_node.get("orderNum"),
        "winner_entrant_id": game_node.get("winnerId"),
        "stage_id": stage.get("id"),
        "stage_name": stage.get("name"),
        "player1_character_id": p1_char_id,
        "player1_character_name": p1_char_name,
        "player2_character_id": p2_char_id,
        "player2_character_name": p2_char_name,
    }


class MeleeCollector:
    """Orchestrates data collection from start.gg into SQLite."""

    def __init__(self, client: StartGGClient, db: MeleeDB):
        self.client = client
        self.db = db

    def _discover_chunk(self, after_date: int, before_date: int) -> list[dict]:
        """Fetch one chunk of tournaments from the API."""
        return self.client.paginate(
            TOURNAMENTS_BY_GAME,
            {
                "videogameId": MELEE_VIDEOGAME_ID,
                "afterDate": after_date,
                "beforeDate": before_date,
                "perPage": 50,
            },
            data_path=["tournaments"],
        )

    def discover_tournaments(
        self,
        after_date: int,
        before_date: int,
        min_attendees: int = 50,
    ) -> list[dict]:
        """Find Melee tournaments in a date range.

        Automatically splits into 3-month windows to stay under the
        start.gg 10,000-entry pagination limit.

        Args:
            after_date: Unix timestamp for the start of the range.
            before_date: Unix timestamp for the end of the range.
            min_attendees: Minimum number of attendees to include.

        Returns:
            List of tournament dicts from the API, filtered to relevant ones.
        """
        # Split into 3-month (90-day) windows to avoid the 10k limit
        chunk_size = 90 * 24 * 60 * 60  # 90 days in seconds
        all_tournaments = []
        seen_ids = set()

        window_start = after_date
        while window_start < before_date:
            window_end = min(window_start + chunk_size, before_date)
            logger.info(f"Discovering tournaments: window {window_start} to {window_end}")

            chunk = self._discover_chunk(window_start, window_end)

            for t in chunk:
                if t["id"] not in seen_ids:
                    seen_ids.add(t["id"])
                    all_tournaments.append(t)

            logger.info(f"  Found {len(chunk)} in this window ({len(all_tournaments)} total)")
            window_start = window_end

        # Filter to offline tournaments with enough attendees
        filtered = []
        for t in all_tournaments:
            attendees = t.get("numAttendees") or 0
            is_online = t.get("isOnline", False)

            if attendees >= min_attendees and not is_online:
                filtered.append(t)

        logger.info(
            f"Found {len(filtered)} tournaments with {min_attendees}+ attendees "
            f"(from {len(all_tournaments)} total)"
        )
        return filtered

    def collect_tournament(self, tournament: dict, skip_existing: bool = True):
        """Collect all set/game data for a tournament and store in the DB.

        Args:
            tournament: Tournament dict from the API (with nested events).
            skip_existing: If True, skip events that already have sets in the DB.
        """
        t_id = tournament["id"]
        t_name = tournament["name"]

        # Store the tournament itself
        self.db.upsert_tournament(tournament)

        # Find Melee singles events
        melee_events = []
        for event in tournament.get("events") or []:
            vg = event.get("videogame") or {}
            if vg.get("id") == MELEE_VIDEOGAME_ID:
                melee_events.append(event)

        if not melee_events:
            logger.warning(f"No Melee events found for {t_name}")
            return

        for event in melee_events:
            e_id = event["id"]
            e_name = event["name"]

            self.db.upsert_event(event, t_id)

            if skip_existing and self.db.event_has_sets(e_id):
                logger.info(f"  Skipping {e_name} (already collected)")
                continue

            logger.info(f"  Collecting sets for: {t_name} — {e_name}")

            set_nodes = self.client.paginate(
                EVENT_SETS,
                {"eventId": e_id, "perPage": 15},
                data_path=["event", "sets"],
            )

            set_count = 0
            game_count = 0

            for set_node in set_nodes:
                set_record, game_records = _parse_set(set_node, e_id)
                if set_record is None:
                    continue

                # Upsert both players
                slots = set_node.get("slots") or []
                for slot in slots:
                    entrant = slot.get("entrant") or {}
                    for participant in entrant.get("participants") or []:
                        player = participant.get("player") or {}
                        if player.get("id") and player.get("gamerTag"):
                            self.db.upsert_player(player["id"], player["gamerTag"])

                self.db.upsert_set(set_record)
                set_count += 1

                for game_record in game_records:
                    self.db.upsert_game(game_record)
                    game_count += 1

            self.db.commit()
            logger.info(f"    Stored {set_count} sets, {game_count} games")

    def collect_date_range(
        self,
        after_date: int,
        before_date: int,
        min_attendees: int = 50,
        skip_existing: bool = True,
    ):
        """Full pipeline: discover tournaments and collect all their data.

        Args:
            after_date: Unix timestamp for start of range.
            before_date: Unix timestamp for end of range.
            min_attendees: Minimum attendees for a tournament to be included.
            skip_existing: Skip events that already have data in the DB.
        """
        tournaments = self.discover_tournaments(after_date, before_date, min_attendees)

        for tournament in tqdm(tournaments, desc="Tournaments"):
            try:
                self.collect_tournament(tournament, skip_existing=skip_existing)
            except Exception:
                logger.exception(f"Failed to collect {tournament['name']}")
                continue
