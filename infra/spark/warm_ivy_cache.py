"""Resolve spark.jars.packages at image build so jobs start without downloading jars."""

from pyspark.sql import SparkSession

SparkSession.builder.master("local[1]").appName("warm-ivy-cache").getOrCreate().stop()
