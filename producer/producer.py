"""
Kafka Producer for Transaction Frequency Streaming

Reads synthetic transaction data, filters by network/transaction type combo,
aggregates into hourly transaction counts, and streams to a Kafka topic
for real-time FCVAE anomaly detection.

Aggregation logic matches TransactionPreprocessor.load_and_aggregate():
  - Bin timestamps to hour with dt.floor("h")
  - Group by hour and count transactions
  - Fill missing hours with 0
"""

import os
import sys
import time
import json
import logging
from datetime import datetime

import pandas as pd
from confluent_kafka import Producer, KafkaError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def delivery_callback(err, msg):
    """Callback for message delivery reports."""
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] @ {msg.offset()}")


def create_producer(bootstrap_servers: str) -> Producer:
    """Create and configure the Kafka producer."""
    config = {
        "bootstrap.servers": bootstrap_servers,
        "client.id": "transaction-producer",
        "acks": "all",
        "retries": 3,
        "retry.backoff.ms": 1000,
        "linger.ms": 5,
        "batch.size": 16384,
    }
    return Producer(config)


def load_and_aggregate(file_path: str, network_type: str, txn_type: str, split: str) -> pd.DataFrame:
    """
    Load transaction data, filter by combo and split, aggregate to hourly counts.

    Mirrors the aggregation in TransactionPreprocessor.load_and_aggregate():
    bin to hour, group, count, fill gaps with 0.

    Returns DataFrame with columns [timestamp, value] where value is hourly count.
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df)} total records")

    # Filter by split
    if "split" in df.columns:
        df = df[df["split"] == split]
        logger.info(f"Filtered to '{split}' split: {len(df)} records")

    # Filter by combo
    df = df[(df["network_type"] == network_type) & (df["transaction_type"] == txn_type)]
    logger.info(f"Filtered to {network_type}/{txn_type}: {len(df)} records")

    if df.empty:
        raise ValueError(
            f"No data found for combo {network_type}/{txn_type} in split '{split}'. "
            f"Check COMBO and DATA_SPLIT environment variables."
        )

    # Aggregate to hourly transaction counts (same as TransactionPreprocessor)
    df["hour_bucket"] = df["timestamp"].dt.floor("h")
    hourly_counts = df.groupby("hour_bucket").agg(
        value=("timestamp", "size"),
        is_anomaly=("is_anomaly", "max")
    ).reset_index()

    # Fill missing hours with 0 (complete hourly index)
    full_hours = pd.date_range(
        start=hourly_counts["hour_bucket"].min(),
        end=hourly_counts["hour_bucket"].max(),
        freq="h",
    )
    full_df = pd.DataFrame({"hour_bucket": full_hours})
    hourly_counts = full_df.merge(hourly_counts, on="hour_bucket", how="left")
    hourly_counts["value"] = hourly_counts["value"].fillna(0).astype(int)
    hourly_counts["is_anomaly"] = hourly_counts["is_anomaly"].fillna(0).astype(int)
    hourly_counts = hourly_counts.rename(columns={"hour_bucket": "timestamp"})

    logger.info(f"Aggregated to {len(hourly_counts)} hourly counts")
    logger.info(f"  Date range: {hourly_counts['timestamp'].iloc[0]} to {hourly_counts['timestamp'].iloc[-1]}")
    logger.info(f"  Value range: [{hourly_counts['value'].min()}, {hourly_counts['value'].max()}]")

    return hourly_counts


def stream_data(
    producer: Producer,
    topic: str,
    df: pd.DataFrame,
    delay_seconds: float = 0.1,
    loop: bool = True,
):
    """Stream hourly transaction counts to Kafka topic.

    When looping, timestamps are offset on each iteration so they continue
    monotonically from where the previous iteration ended (+ 1 hour gap).
    """
    from datetime import timedelta

    logger.info(f"Starting to stream {len(df)} hourly records to topic '{topic}'")
    logger.info(f"Delay between messages: {delay_seconds}s, Loop: {loop}")

    iteration = 0
    total_messages = 0
    time_offset = timedelta(0)

    # Compute the span of the original data for offsetting on loop
    data_span = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0] + timedelta(hours=1)

    while True:
        iteration += 1
        logger.info(f"Starting iteration {iteration} (time_offset={time_offset})")

        for idx, row in df.iterrows():
            timestamp = row["timestamp"] + time_offset
            message = {
                "timestamp": timestamp.isoformat(),
                "value": int(row["value"]),
                "is_anomaly": int(row["is_anomaly"]),
                "produced_at": datetime.utcnow().isoformat(),
                "sequence_id": total_messages
            }

            try:
                producer.produce(
                    topic=topic,
                    key=str(total_messages).encode("utf-8"),
                    value=json.dumps(message).encode("utf-8"),
                    callback=delivery_callback
                )
                total_messages += 1

                if total_messages % 100 == 0:
                    logger.info(f"Produced {total_messages} messages")
                    producer.flush()

                producer.poll(0)
                time.sleep(delay_seconds)

            except BufferError:
                logger.warning("Producer queue full, waiting...")
                producer.flush()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error producing message: {e}")
                raise

        producer.flush()
        logger.info(f"Completed iteration {iteration}, total messages: {total_messages}")

        if not loop:
            break

        # Offset timestamps for the next iteration so they continue monotonically
        time_offset += data_span

    logger.info(f"Streaming complete. Total messages produced: {total_messages}")


def wait_for_kafka(bootstrap_servers: str, max_retries: int = 30, retry_interval: int = 2):
    """Wait for Kafka to be available before starting."""
    logger.info(f"Waiting for Kafka at {bootstrap_servers}...")

    for attempt in range(max_retries):
        try:
            producer = Producer({"bootstrap.servers": bootstrap_servers})
            metadata = producer.list_topics(timeout=5)
            logger.info(f"Connected to Kafka. Available topics: {list(metadata.topics.keys())}")
            return True
        except Exception as e:
            logger.warning(f"Kafka not ready (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(retry_interval)

    raise RuntimeError(f"Could not connect to Kafka after {max_retries} attempts")


def wait_for_app_ready(app_url: str, max_retries: int = 60, retry_interval: int = 2):
    """Wait for the app's Spark streaming to be ready before producing."""
    import urllib.request
    import urllib.error

    health_url = f"{app_url}/health"
    logger.info(f"Waiting for app to be ready at {health_url}...")

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                data = json.loads(response.read().decode())
                if data.get("spark_ready", False):
                    logger.info(f"App is ready! Spark streaming connected.")
                    return True
                else:
                    logger.info(f"App starting... (attempt {attempt + 1}/{max_retries})")
        except urllib.error.URLError as e:
            logger.debug(f"App not reachable yet (attempt {attempt + 1}/{max_retries}): {e}")
        except Exception as e:
            logger.debug(f"Health check failed (attempt {attempt + 1}/{max_retries}): {e}")

        time.sleep(retry_interval)

    logger.warning(f"App did not become ready after {max_retries} attempts, proceeding anyway")
    return False


def parse_combo(combo_str: str):
    """Parse combo string like 'Accel_CMP' into (network_type, txn_type)."""
    parts = combo_str.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid COMBO format: '{combo_str}'. Expected 'Network_TxnType' (e.g., 'Accel_CMP')")
    network = parts[0]
    txn_type = "no-pin" if "nopin" in parts[1].lower() else parts[1]
    return network, txn_type


def main():
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "anomaly_stream")
    data_path = os.getenv("DATA_PATH", "/app/data/synthetic_transactions_v2_split60.csv")
    delay = float(os.getenv("MESSAGE_DELAY_SECONDS", "0.1"))
    loop_data = os.getenv("LOOP_DATA", "true").lower() == "true"
    combo_str = os.getenv("COMBO", "Accel_nopin")
    split = os.getenv("DATA_SPLIT", "test")
    app_url = os.getenv("APP_URL", "http://app:8050")
    wait_for_app = os.getenv("WAIT_FOR_APP", "true").lower() == "true"

    network_type, txn_type = parse_combo(combo_str)

    logger.info("=" * 50)
    logger.info("Transaction Frequency Kafka Producer")
    logger.info("=" * 50)
    logger.info(f"Bootstrap Servers: {bootstrap_servers}")
    logger.info(f"Topic: {topic}")
    logger.info(f"Data Path: {data_path}")
    logger.info(f"Combo: {network_type}/{txn_type}")
    logger.info(f"Split: {split}")
    logger.info(f"Message Delay: {delay}s")
    logger.info(f"Loop Data: {loop_data}")
    logger.info(f"Wait for App: {wait_for_app}")
    logger.info("=" * 50)

    wait_for_kafka(bootstrap_servers)

    if wait_for_app:
        wait_for_app_ready(app_url)

    producer = create_producer(bootstrap_servers)
    df = load_and_aggregate(data_path, network_type, txn_type, split)

    try:
        stream_data(
            producer, topic, df,
            delay_seconds=delay,
            loop=loop_data,
        )
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        logger.info("Flushing remaining messages...")
        producer.flush(timeout=10)
        logger.info("Producer shutdown complete")


if __name__ == "__main__":
    main()
