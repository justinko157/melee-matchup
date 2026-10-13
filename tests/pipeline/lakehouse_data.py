"""Write the silver columns gold reads into a PyIceberg catalog (SqlCatalog in tests)."""

from datetime import UTC, datetime

import pyarrow as pa

TS = pa.timestamp("us", tz="UTC")
SILVER_SCHEMAS = {
    "silver.sets": pa.schema(
        [
            ("set_id", pa.int64()),
            ("event_id", pa.int64()),
            ("completed_at", TS),
            ("player1_id", pa.int64()),
            ("player2_id", pa.int64()),
            ("winner_player_id", pa.int64()),
            ("player1_score", pa.float64()),
            ("player2_score", pa.float64()),
            ("player1_seed", pa.int64()),
            ("player2_seed", pa.int64()),
        ]
    ),
    "silver.events": pa.schema([("event_id", pa.int64()), ("tournament_id", pa.int64())]),
    "silver.tournaments": pa.schema(
        [("tournament_id", pa.int64()), ("name", pa.string()), ("num_attendees", pa.int64())]
    ),
    "silver.players": pa.schema([("player_id", pa.int64()), ("gamer_tag", pa.string())]),
}


def silver_set(set_id, p1, p2, winner, ts, event_id=10, seeds=(1, 2)):
    return {
        "set_id": set_id,
        "event_id": event_id,
        "completed_at": datetime.fromtimestamp(ts, UTC) if ts is not None else None,
        "player1_id": p1,
        "player2_id": p2,
        "winner_player_id": winner,
        "player1_score": 2.0,
        "player2_score": 1.0,
        "player1_seed": seeds[0],
        "player2_seed": seeds[1],
    }


def write_silver(catalog, *, sets=(), events=(), tournaments=(), players=()):
    catalog.create_namespace_if_not_exists("silver")
    for identifier, rows in (
        ("silver.sets", sets),
        ("silver.events", events),
        ("silver.tournaments", tournaments),
        ("silver.players", players),
    ):
        schema = SILVER_SCHEMAS[identifier]
        table = catalog.create_table_if_not_exists(identifier, schema=schema)
        if rows:
            table.append(pa.Table.from_pylist(list(rows), schema=schema))
