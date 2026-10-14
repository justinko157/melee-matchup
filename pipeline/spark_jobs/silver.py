"""Pure transforms from bronze rows to silver rows, plus silver DDL and MERGE SQL.

Parsing rules come from the retired src/collector.py:
- a set needs two slots, both with a player, and state 3;
- the winner is resolved from winnerId to the matching entrant's player.

Tournaments in EXCLUDED_TOURNAMENT_IDS are kept out of silver entirely.
"""

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DataType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from pipeline.ingest.startgg import MELEE_VIDEOGAME_ID

SILVER_NAMESPACE = "silver"
WATERMARKS_TABLE = f"{SILVER_NAMESPACE}._watermarks"
MERGE_ORDER = ("tournaments", "events", "players", "sets", "games")
KEYS = {
    "tournaments": "tournament_id",
    "events": "event_id",
    "players": "player_id",
    "sets": "set_id",
    "games": "game_id",
}
LINEAGE_COLUMNS = ("source_ingested_at", "source_snapshot_id", "silver_updated_at")
ORDER_COLUMNS = ("source_ingested_at", "kafka_partition", "kafka_offset")
DROP_REASONS = ("too_few_slots", "missing_player", "not_completed", "unresolved_winner")
# Not a data-quality drop: excluded sets are counted apart from DROP_REASONS.
EXCLUDED = "excluded_tournament"
# Fabricated brackets. All five claim the same in-person weekend (2020-12-12, mid-COVID),
# yet their 51k sets were completed 2021-06..2022-08 by 4,307 players who appear in no
# other tournament: "Great Idea (Not)" parts 3-5, "Melee Replace idea" and "Regional
# Battles, for Fun and stuff".
EXCLUDED_TOURNAMENT_IDS = (294173, 315801, 329733, 416947, 439378)

SCHEMAS: dict[str, tuple[tuple[str, str], ...]] = {
    "tournaments": (
        ("tournament_id", "BIGINT"),
        ("name", "STRING"),
        ("slug", "STRING"),
        ("num_attendees", "BIGINT"),
        ("start_at", "TIMESTAMP"),
        ("end_at", "TIMESTAMP"),
        ("city", "STRING"),
        ("state", "STRING"),
        ("country_code", "STRING"),
        ("is_online", "BOOLEAN"),
    ),
    "events": (
        ("event_id", "BIGINT"),
        ("tournament_id", "BIGINT"),
        ("name", "STRING"),
        ("slug", "STRING"),
        ("num_entrants", "BIGINT"),
    ),
    "players": (("player_id", "BIGINT"), ("gamer_tag", "STRING")),
    "sets": (
        ("set_id", "BIGINT"),
        ("event_id", "BIGINT"),
        ("round", "BIGINT"),
        ("full_round_text", "STRING"),
        ("phase_name", "STRING"),
        ("completed_at", "TIMESTAMP"),
        ("total_games", "BIGINT"),
        ("display_score", "STRING"),
        ("entrant1_id", "BIGINT"),
        ("entrant2_id", "BIGINT"),
        ("player1_id", "BIGINT"),
        ("player2_id", "BIGINT"),
        ("winner_player_id", "BIGINT"),
        ("player1_score", "DOUBLE"),
        ("player2_score", "DOUBLE"),
        ("player1_seed", "BIGINT"),
        ("player2_seed", "BIGINT"),
    ),
    "games": (
        ("game_id", "BIGINT"),
        ("set_id", "BIGINT"),
        ("game_number", "BIGINT"),
        ("winner_entrant_id", "BIGINT"),
        ("stage_id", "BIGINT"),
        ("stage_name", "STRING"),
        ("player1_character_id", "BIGINT"),
        ("player1_character_name", "STRING"),
        ("player2_character_id", "BIGINT"),
        ("player2_character_name", "STRING"),
        ("completed_at", "TIMESTAMP"),
    ),
}
PARTITIONS = {"sets": "months(completed_at)", "games": "months(completed_at)"}
# v2 + copy-on-write keeps the tables readable by PyIceberg without delete files.
TABLE_PROPERTIES = (
    "TBLPROPERTIES ('format-version'='2', 'write.merge.mode'='copy-on-write', "
    "'write.update.mode'='copy-on-write', 'write.delete.mode'='copy-on-write')"
)


def table_name(entity: str) -> str:
    return f"{SILVER_NAMESPACE}.{entity}"


def business_columns(entity: str) -> tuple[str, ...]:
    return tuple(name for name, _ in SCHEMAS[entity])


def silver_columns(entity: str) -> tuple[str, ...]:
    return business_columns(entity) + LINEAGE_COLUMNS


def silver_table_ddl(entity: str) -> str:
    columns = ", ".join(f"{name} {kind}" for name, kind in SCHEMAS[entity])
    lineage = "source_ingested_at TIMESTAMP, source_snapshot_id BIGINT, silver_updated_at TIMESTAMP"
    partition = f"PARTITIONED BY ({PARTITIONS[entity]})" if entity in PARTITIONS else ""
    return (
        f"CREATE TABLE IF NOT EXISTS {table_name(entity)} ({columns}, {lineage}) "
        f"USING iceberg {partition} {TABLE_PROPERTIES}"
    )


WATERMARKS_DDL = (
    f"CREATE TABLE IF NOT EXISTS {WATERMARKS_TABLE} "
    f"(bronze_table STRING, snapshot_id BIGINT, updated_at TIMESTAMP) "
    f"USING iceberg {TABLE_PROPERTIES}"
)


def _struct(**fields: DataType) -> StructType:
    return StructType([StructField(name, kind) for name, kind in fields.items()])


EVENT_NODE = _struct(
    id=LongType(),
    name=StringType(),
    slug=StringType(),
    numEntrants=LongType(),
    videogame=_struct(id=LongType()),
)
TOURNAMENT_PAYLOAD = _struct(
    id=LongType(),
    name=StringType(),
    slug=StringType(),
    numAttendees=LongType(),
    startAt=LongType(),
    endAt=LongType(),
    city=StringType(),
    addrState=StringType(),
    countryCode=StringType(),
    isOnline=BooleanType(),
    events=ArrayType(EVENT_NODE),
)
ENTRANT = _struct(
    id=LongType(),
    name=StringType(),
    initialSeedNum=LongType(),
    participants=ArrayType(_struct(player=_struct(id=LongType(), gamerTag=StringType()))),
)
SLOT = _struct(
    seed=_struct(seedNum=LongType()),
    standing=_struct(placement=LongType(), stats=_struct(score=_struct(value=DoubleType()))),
    entrant=ENTRANT,
)
GAME = _struct(
    id=LongType(),
    orderNum=LongType(),
    winnerId=LongType(),
    stage=_struct(id=LongType(), name=StringType()),
    selections=ArrayType(
        _struct(
            entrant=_struct(id=LongType()),
            character=_struct(id=LongType(), name=StringType()),
        )
    ),
)
# The set id is a string because unplayed sets have ids like "preview_123_1".
SET_PAYLOAD = _struct(
    id=StringType(),
    displayScore=StringType(),
    winnerId=LongType(),
    round=LongType(),
    fullRoundText=StringType(),
    completedAt=LongType(),
    totalGames=LongType(),
    state=LongType(),
    games=ArrayType(GAME),
    slots=ArrayType(SLOT),
    phaseGroup=_struct(phase=_struct(name=StringType())),
)


def _nth(array: Column, n: int) -> Column:
    # try_element_at returns null past the end; plain indexing errors under ANSI mode.
    return F.try_element_at(array, F.lit(n))


def _source(bronze: DataFrame, payload_schema: StructType) -> list[Column]:
    return [
        F.col("ingested_at").alias("source_ingested_at"),
        F.col("kafka_partition"),
        F.col("kafka_offset"),
        F.from_json(F.col("payload"), payload_schema).alias("p"),
    ]


def _is_excluded_tournament(tournament_id: Column) -> Column:
    return tournament_id.isin(list(EXCLUDED_TOURNAMENT_IDS))


def parse_tournaments(bronze: DataFrame) -> DataFrame:
    """One typed row per bronze tournament row (duplicates kept), minus excluded tournaments."""
    parsed = bronze.select(*_source(bronze, TOURNAMENT_PAYLOAD))
    return parsed.where(F.col("p.id").isNotNull() & ~_is_excluded_tournament(F.col("p.id")))


def excluded_event_ids(bronze: DataFrame) -> DataFrame:
    """Distinct event ids (column event_id) of the excluded tournaments in bronze."""
    parsed = bronze.select(*_source(bronze, TOURNAMENT_PAYLOAD))
    events = parsed.where(_is_excluded_tournament(F.col("p.id"))).select(
        F.explode(F.col("p.events")).alias("e")
    )
    return events.select(F.col("e.id").alias("event_id")).where("event_id IS NOT NULL").distinct()


def tournament_rows(parsed: DataFrame) -> DataFrame:
    return parsed.select(
        F.col("p.id").alias("tournament_id"),
        F.col("p.name").alias("name"),
        F.col("p.slug").alias("slug"),
        F.col("p.numAttendees").alias("num_attendees"),
        F.timestamp_seconds(F.col("p.startAt")).alias("start_at"),
        F.timestamp_seconds(F.col("p.endAt")).alias("end_at"),
        F.col("p.city").alias("city"),
        F.col("p.addrState").alias("state"),
        F.col("p.countryCode").alias("country_code"),
        F.col("p.isOnline").alias("is_online"),
        *ORDER_COLUMNS,
    )


def event_rows(parsed: DataFrame) -> DataFrame:
    """Melee events from each tournament payload's events array."""
    events = parsed.select(
        F.col("p.id").alias("tournament_id"),
        F.explode(F.col("p.events")).alias("e"),
        *ORDER_COLUMNS,
    )
    return events.where(
        F.col("e.id").isNotNull() & (F.col("e.videogame.id") == F.lit(MELEE_VIDEOGAME_ID))
    ).select(
        F.col("e.id").alias("event_id"),
        "tournament_id",
        F.col("e.name").alias("name"),
        F.col("e.slug").alias("slug"),
        F.col("e.numEntrants").alias("num_entrants"),
        *ORDER_COLUMNS,
    )


def parse_sets(bronze: DataFrame, excluded_events: tuple[int, ...] = ()) -> DataFrame:
    """One row per bronze set row with typed columns and a drop_reason (null = keep).

    Sets in excluded_events get the EXCLUDED reason ahead of any quality rule.
    """
    event_id = (
        F.get_json_object(F.col("query_variables"), "$.eventId").try_cast("long").alias("event_id")
    )
    df = bronze.select(*_source(bronze, SET_PAYLOAD), event_id)
    slots = F.col("p.slots")
    slot1, slot2 = _nth(slots, 1), _nth(slots, 2)
    entrant1, entrant2 = slot1["entrant"], slot2["entrant"]
    player1 = _nth(entrant1["participants"], 1)["player"]
    player2 = _nth(entrant2["participants"], 1)["player"]
    set_id = F.col("p.id").try_cast("long")
    winner = F.col("p.winnerId")
    excluded = F.col("event_id").isin(list(excluded_events)) if excluded_events else F.lit(False)
    drop_reason = (
        F.when(excluded, EXCLUDED)
        .when(F.coalesce(F.size(slots), F.lit(0)) < 2, "too_few_slots")
        .when(player1["id"].isNull() | player2["id"].isNull(), "missing_player")
        .when(set_id.isNull() | ~F.coalesce(F.col("p.state") == 3, F.lit(False)), "not_completed")
        .when(
            ~F.coalesce((winner == entrant1["id"]) | (winner == entrant2["id"]), F.lit(False)),
            "unresolved_winner",
        )
    )
    return df.select(
        *ORDER_COLUMNS,
        set_id.alias("set_id"),
        "event_id",
        F.col("p.round").alias("round"),
        F.col("p.fullRoundText").alias("full_round_text"),
        F.col("p.phaseGroup.phase.name").alias("phase_name"),
        F.timestamp_seconds(F.col("p.completedAt")).alias("completed_at"),
        F.col("p.totalGames").alias("total_games"),
        F.col("p.displayScore").alias("display_score"),
        entrant1["id"].alias("entrant1_id"),
        entrant2["id"].alias("entrant2_id"),
        player1["id"].alias("player1_id"),
        player2["id"].alias("player2_id"),
        F.when(winner == entrant1["id"], player1["id"])
        .when(winner == entrant2["id"], player2["id"])
        .alias("winner_player_id"),
        slot1["standing"]["stats"]["score"]["value"].alias("player1_score"),
        slot2["standing"]["stats"]["score"]["value"].alias("player2_score"),
        F.coalesce(slot1["seed"]["seedNum"], entrant1["initialSeedNum"]).alias("player1_seed"),
        F.coalesce(slot2["seed"]["seedNum"], entrant2["initialSeedNum"]).alias("player2_seed"),
        player1["gamerTag"].alias("player1_tag"),
        player2["gamerTag"].alias("player2_tag"),
        F.col("p.games").alias("games"),
        drop_reason.alias("drop_reason"),
    )


def _kept(parsed: DataFrame) -> DataFrame:
    return parsed.where(F.col("drop_reason").isNull())


def set_rows(parsed: DataFrame) -> DataFrame:
    return _kept(parsed).select(*business_columns("sets"), *ORDER_COLUMNS)


def player_rows(parsed: DataFrame) -> DataFrame:
    kept = _kept(parsed)
    return kept.select(
        F.col("player1_id").alias("player_id"),
        F.col("player1_tag").alias("gamer_tag"),
        *ORDER_COLUMNS,
    ).unionByName(
        kept.select(
            F.col("player2_id").alias("player_id"),
            F.col("player2_tag").alias("gamer_tag"),
            *ORDER_COLUMNS,
        )
    )


def _character(entrant_column: str) -> Column:
    matches = F.filter(F.col("g.selections"), lambda s: s["entrant"]["id"] == F.col(entrant_column))
    return _nth(matches, 1)["character"]


def game_rows(parsed: DataFrame) -> DataFrame:
    games = _kept(parsed).select(
        "set_id",
        "completed_at",
        "entrant1_id",
        "entrant2_id",
        *ORDER_COLUMNS,
        F.explode(F.col("games")).alias("g"),
    )
    character1, character2 = _character("entrant1_id"), _character("entrant2_id")
    return games.where(F.col("g.id").isNotNull()).select(
        F.col("g.id").alias("game_id"),
        "set_id",
        F.col("g.orderNum").alias("game_number"),
        F.col("g.winnerId").alias("winner_entrant_id"),
        F.col("g.stage.id").alias("stage_id"),
        F.col("g.stage.name").alias("stage_name"),
        character1["id"].alias("player1_character_id"),
        character1["name"].alias("player1_character_name"),
        character2["id"].alias("player2_character_id"),
        character2["name"].alias("player2_character_name"),
        "completed_at",
        *ORDER_COLUMNS,
    )


def drop_counts(parsed: DataFrame) -> tuple[int, dict[str, int], int]:
    """(bronze set rows read, dropped rows per reason, excluded rows), in one pass."""
    counts = {
        row.drop_reason: row["count"] for row in parsed.groupBy("drop_reason").count().collect()
    }
    total = sum(counts.values())
    dropped = {reason: int(counts.get(reason, 0)) for reason in DROP_REASONS}
    return total, dropped, int(counts.get(EXCLUDED, 0))


def newest_per_key(df: DataFrame, key: str) -> DataFrame:
    """Keep the newest version of each key: latest ingest, then highest Kafka position."""
    order = [F.col(column).desc() for column in ORDER_COLUMNS]
    ranked = df.withColumn("_rank", F.row_number().over(Window.partitionBy(key).orderBy(*order)))
    return ranked.where(F.col("_rank") == 1).drop("_rank")


def finalize(df: DataFrame, entity: str, snapshot_id: int) -> DataFrame:
    """Silver column order, with the bronze snapshot the batch was read up to."""
    return df.select(
        *business_columns(entity),
        "source_ingested_at",
        F.lit(snapshot_id).cast("bigint").alias("source_snapshot_id"),
        F.current_timestamp().alias("silver_updated_at"),
    )


def merge_sql(entity: str, view: str) -> str:
    """Upsert one staged batch. Only an equal-or-newer bronze version overwrites a row."""
    key = KEYS[entity]
    return f"""
        MERGE INTO {table_name(entity)} AS t
        USING {view} AS s
        ON t.{key} = s.{key}
        WHEN MATCHED AND s.source_ingested_at >= t.source_ingested_at THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """
