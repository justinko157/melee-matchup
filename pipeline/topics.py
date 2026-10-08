"""Kafka topic names shared by producers, the Spark job and Compose."""

TOURNAMENTS_TOPIC = "startgg.tournaments.raw"
SETS_TOPIC = "startgg.sets.raw"
DLQ_TOPIC = "startgg.ingest.dlq"

RAW_TOPICS = (TOURNAMENTS_TOPIC, SETS_TOPIC)
ALL_TOPICS = (*RAW_TOPICS, DLQ_TOPIC)
