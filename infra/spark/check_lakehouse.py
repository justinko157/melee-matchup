"""Check Spark -> Lakekeeper -> Silo end to end: create, write and read a table."""

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("check-lakehouse").getOrCreate()
spark.sql("CREATE NAMESPACE IF NOT EXISTS healthcheck")
spark.sql("CREATE OR REPLACE TABLE healthcheck.ping (id INT, note STRING) USING iceberg")
spark.sql("INSERT INTO healthcheck.ping VALUES (1, 'ok')")
rows = [tuple(r) for r in spark.sql("SELECT id, note FROM healthcheck.ping").collect()]
assert rows == [(1, "ok")], rows
print("spark lakehouse check ok:", rows)
spark.stop()
