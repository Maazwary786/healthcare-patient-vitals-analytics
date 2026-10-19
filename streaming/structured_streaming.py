
"""Consume synthetic vitals from Kafka using Spark Structured Streaming."""

import logging
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    from_json,
    max as spark_max,
    min as spark_min,
    sum as spark_sum,
    to_timestamp,
    window,
    when,
)
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)


ROOT = Path(__file__).resolve().parents[1]
STREAM_ROOT = ROOT / "data" / "streaming"
BRONZE_PATH = STREAM_ROOT / "bronze" / "vitals"
BRONZE_CHECKPOINT = STREAM_ROOT / "checkpoints" / "bronze"
ALERT_CHECKPOINT = STREAM_ROOT / "checkpoints" / "alerts"

BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
)
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "vitals_stream")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


EVENT_SCHEMA = StructType([
    StructField("event_id", StringType(), True),
    StructField("patient_key", StringType(), True),
    StructField("event_time", StringType(), True),
    StructField("heart_rate", DoubleType(), True),
    StructField("respiratory_rate", DoubleType(), True),
    StructField("spo2", DoubleType(), True),
    StructField("source", StringType(), True),
])


def main() -> None:
    STREAM_ROOT.mkdir(parents=True, exist_ok=True)
    BRONZE_PATH.mkdir(parents=True, exist_ok=True)
    BRONZE_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    ALERT_CHECKPOINT.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("HealthcareVitalsStreaming")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    kafka_events = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        kafka_events
        .selectExpr(
            "CAST(key AS STRING) AS kafka_key",
            "CAST(value AS STRING) AS raw_json",
            "topic",
            "partition",
            "offset",
            "timestamp AS kafka_timestamp",
        )
        .withColumn("event", from_json(col("raw_json"), EVENT_SCHEMA))
        .select(
            "kafka_key",
            "raw_json",
            "topic",
            "partition",
            "offset",
            "kafka_timestamp",
            col("event.event_id").alias("event_id"),
            col("event.patient_key").alias("patient_key"),
            to_timestamp(
                col("event.event_time"),
                "yyyy-MM-dd'T'HH:mm:ss.SSSX",
            ).alias("event_time"),
            col("event.heart_rate").alias("heart_rate"),
            col("event.respiratory_rate").alias("respiratory_rate"),
            col("event.spo2").alias("spo2"),
            col("event.source").alias("source"),
        )
    )

    valid = parsed.filter(
        col("event_id").isNotNull()
        & col("patient_key").isNotNull()
        & col("event_time").isNotNull()
        & col("source").isNotNull()
        & col("heart_rate").between(1.0, 300.0)
        & col("respiratory_rate").between(1.0, 100.0)
        & col("spo2").between(0.0, 100.0)
    )

    # Query 1: persist validated events as Parquet Bronze data.
    bronze_query = (
        valid.writeStream
        .format("parquet")
        .outputMode("append")
        .option("path", str(BRONZE_PATH))
        .option("checkpointLocation", str(BRONZE_CHECKPOINT))
        .trigger(processingTime="2 seconds")
        .start()
    )

    # Query 2: evaluate overlapping event-time windows.
    # Alert is emitted only if every sample in a qualifying window is low.
    low_events = (
        valid
        .withWatermark("event_time", "5 seconds")
        .groupBy(
            "patient_key",
            window(col("event_time"), "190 seconds", "10 seconds"),
        )
        .agg(
            count("*").alias("sample_count"),
            spark_min("event_time").alias("first_event_time"),
            spark_max("event_time").alias("last_event_time"),
            spark_sum(
                when(col("spo2") < 90.0, 1).otherwise(0)
            ).alias("low_sample_count"),
        )
        .withColumn(
            "observed_span_seconds",
            col("last_event_time").cast("long")
            - col("first_event_time").cast("long"),
        )
        .filter(
            (col("observed_span_seconds") >= 180)
            & (col("sample_count") >= 60)
            & (col("low_sample_count") == col("sample_count"))
        )
        .select(
            "patient_key",
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            "sample_count",
            "observed_span_seconds",
        )
    )

    alerts_query = (
        low_events.writeStream
        .format("console")
        .outputMode("update")
        .option("truncate", "false")
        .option("checkpointLocation", str(ALERT_CHECKPOINT))
        .trigger(processingTime="2 seconds")
        .start()
    )

    logger.info("Bronze and alert streaming queries started")

    try:
        spark.streams.awaitAnyTermination()
    finally:
        for query in spark.streams.active:
            query.stop()
        spark.stop()


if __name__ == "__main__":
    main()
