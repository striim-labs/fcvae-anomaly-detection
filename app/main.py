"""
FCVAE Anomaly Detection Dashboard

Real-time anomaly detection on transaction frequency data using
Kafka streaming, Spark Structured Streaming, FCVAE (Frequency-enhanced
Conditional VAE) detection, and Dash visualization.

Key Features:
- 24-hour sliding window detection
- NLL-based scoring (lower = more anomalous)
- Localized anomaly regions (6-hour windows)
- Real-time visualization with Dash
"""

import json
import os
import threading
import time
import logging
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

import dash
from dash.dependencies import Output, Input
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from base_detector import create_detector, BaseDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# FCVAE Configuration
# ============================================
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "24"))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "24"))
MODEL_PATH = os.environ.get("MODEL_PATH", "models/fcvae")
COMBO_STR = os.environ.get("COMBO", "Accel_nopin")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "16"))
DECISION_MODE = os.environ.get("DECISION_MODE", "severity")
SEVERITY_MARGIN = float(os.environ.get("SEVERITY_MARGIN", "0.5"))

# Parse combo string (e.g., "Accel_CMP" -> ("Accel", "CMP"))
def parse_combo(combo_str: str) -> Optional[Tuple[str, str]]:
    """Parse combo string into (network_type, txn_type) tuple."""
    if not combo_str:
        return None
    parts = combo_str.split("_")
    if len(parts) >= 2:
        network = parts[0]
        txn_type = "no-pin" if "nopin" in parts[1].lower() else parts[1]
        return (network, txn_type)
    return None

COMBO = parse_combo(COMBO_STR)

# Load oracle threshold for this combo (if available)
def load_oracle_threshold(model_path: str, combo_str: str) -> Optional[float]:
    """Load oracle threshold from JSON file for the active combo."""
    oracle_file = Path(model_path) / "oracle_thresholds.json"
    if not oracle_file.exists():
        logger.warning(f"Oracle thresholds file not found: {oracle_file}")
        return None
    with open(oracle_file, "r") as f:
        thresholds = json.load(f)
    threshold = thresholds.get(combo_str)
    if threshold is not None:
        logger.info(f"Loaded oracle threshold for {combo_str}: {threshold}")
    else:
        logger.warning(f"No oracle threshold for {combo_str} in {oracle_file}")
    return threshold

ORACLE_THRESHOLD = load_oracle_threshold(MODEL_PATH, COMBO_STR)
if ORACLE_THRESHOLD is not None:
    DECISION_MODE = "last_point"
    logger.info(f"Oracle threshold active -> forcing decision_mode='last_point'")

# Visualization settings
VISUALIZATION_HOURS = int(os.environ.get("VISUALIZATION_HOURS", "24"))  # Show 1 day of history

# Thread-safe data storage
data_lock = threading.Lock()
BUFFER_SIZE = max(VISUALIZATION_HOURS, 200)

data_store = {
    "data": deque(maxlen=BUFFER_SIZE),
    "total_received": 0,
    "last_batch_size": 0,
    "last_update": None,
    "last_detection": None,
    # Sliding window tracking (stride 1)
    "samples_since_last_window": 0,
    "windows_processed": 0,
    # Two-tier anomaly tracking
    "point_anomalies": [],      # Individual anomalous hours {timestamp, value, point_score}
    "pa_caught_anomalies": [],  # Hours caught by PA metric (orange dots)
    "window_anomalies": [],     # Flagged 24h windows {window_start, window_end, num_anomalous, window_score}
    "last_window_anomalous": False,  # For deduplicating consecutive flagged windows
    "_segment_detected": False, # Has the current anomaly segment been detected?
    # Stream status tracking
    "stream_ended": False,
    "spark_ready": False,
}

# How long to wait before considering the stream ended (seconds)
STREAM_TIMEOUT = 10

# Initialize FCVAE detector
detector: BaseDetector = create_detector(
    detector_type="fcvae",
    model_path=MODEL_PATH,
    combo=COMBO,
    window_size=WINDOW_SIZE,
    min_samples=MIN_SAMPLES,
    n_samples=N_SAMPLES,
    decision_mode=DECISION_MODE,
    severity_margin=SEVERITY_MARGIN,
    oracle_threshold=ORACLE_THRESHOLD,
)


def wait_for_kafka(bootstrap_servers: str, max_retries: int = 30, retry_interval: int = 5):
    """Wait for Kafka to be available."""
    from confluent_kafka import Producer

    logger.info(f"Waiting for Kafka at {bootstrap_servers}...")
    for attempt in range(max_retries):
        try:
            producer = Producer({"bootstrap.servers": bootstrap_servers})
            metadata = producer.list_topics(timeout=10)
            logger.info(f"Connected to Kafka. Topics: {list(metadata.topics.keys())}")
            return True
        except Exception as e:
            logger.warning(f"Kafka not ready (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(retry_interval)

    raise RuntimeError(f"Could not connect to Kafka after {max_retries} attempts")


def ensure_kafka_topic(bootstrap_servers: str, topic: str):
    """Pre-create the Kafka topic so Spark doesn't fail on subscribe."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    metadata = admin.list_topics(timeout=10)

    if topic in metadata.topics:
        logger.info(f"Kafka topic '{topic}' already exists")
        return

    logger.info(f"Creating Kafka topic '{topic}'...")
    futures = admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])
    for t, future in futures.items():
        future.result()  # Block until created
        logger.info(f"Created Kafka topic '{t}'")


def start_spark_streaming():
    """Start Spark Structured Streaming to consume from Kafka."""
    kafka_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    kafka_topic = os.environ.get("KAFKA_TOPIC", "anomaly_stream")
    spark_master = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")

    wait_for_kafka(kafka_servers)
    ensure_kafka_topic(kafka_servers, kafka_topic)

    logger.info(f"Starting Spark session connecting to {spark_master}")

    spark = SparkSession.builder \
        .appName("FCVAEAnomalyDetection") \
        .master(spark_master) \
        .config("spark.driver.host", "app") \
        .config("spark.driver.bindAddress", "0.0.0.0") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3") \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Schema matching producer output
    schema = StructType([
        StructField("timestamp", StringType()),
        StructField("value", IntegerType()),
        StructField("is_anomaly", IntegerType()),
        StructField("produced_at", StringType()),
        StructField("sequence_id", IntegerType())
    ])

    logger.info(f"Subscribing to Kafka topic: {kafka_topic}")

    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_servers) \
        .option("subscribe", kafka_topic) \
        .option("startingOffsets", "earliest") \
        .option("maxOffsetsPerTrigger", 50) \
        .load()

    df = df.selectExpr("CAST(value AS STRING)")
    df_parsed = df.select(from_json(col("value"), schema).alias("data")).select("data.*")
    df_parsed = df_parsed.withColumn("timestamp", to_timestamp(col("timestamp")))

    def process_batch(batch_df, batch_id):
        """Process each micro-batch from Spark Streaming."""
        try:
            pandas_df = batch_df.toPandas()

            if pandas_df.empty:
                logger.debug(f"Batch {batch_id}: empty")
                return

            # Log detailed batch info for debugging
            first_ts = pandas_df["timestamp"].iloc[0]
            last_ts = pandas_df["timestamp"].iloc[-1]
            first_seq = pandas_df["sequence_id"].iloc[0]
            last_seq = pandas_df["sequence_id"].iloc[-1]

            logger.info(
                f"Batch {batch_id}: {len(pandas_df)} records, "
                f"seq_id [{first_seq}-{last_seq}], "
                f"timestamps [{first_ts} to {last_ts}]"
            )

            with data_lock:
                for _, row in pandas_df.iterrows():
                    data_store["data"].append({
                        "timestamp": str(row["timestamp"]),
                        "value": row["value"],
                        "is_anomaly": int(row.get("is_anomaly", 0)),
                        "nll_score": None,
                        "produced_at": row["produced_at"],
                        "sequence_id": row["sequence_id"]
                    })

                data_store["total_received"] += len(pandas_df)
                data_store["last_batch_size"] = len(pandas_df)
                data_store["last_update"] = pd.Timestamp.now().isoformat()
                data_store["samples_since_last_window"] += len(pandas_df)

                logger.info(
                    f"  -> Buffer after batch: {len(data_store['data'])} samples, "
                    f"samples_since_last: {data_store['samples_since_last_window']}"
                )

        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")

    query = df_parsed.writeStream \
        .trigger(processingTime="1 seconds") \
        .foreachBatch(process_batch) \
        .start()

    # Signal that Spark is ready to receive data
    with data_lock:
        data_store["spark_ready"] = True
    logger.info("Spark Streaming started - ready for data")

    query.awaitTermination()


def check_stream_ended():
    """Check if stream has ended (no new data for STREAM_TIMEOUT seconds)."""
    with data_lock:
        last_update = data_store["last_update"]
        if last_update is None:
            return False

        last_update_time = pd.Timestamp(last_update)
        time_since_update = (pd.Timestamp.now() - last_update_time).total_seconds()

        if time_since_update > STREAM_TIMEOUT and not data_store["stream_ended"]:
            data_store["stream_ended"] = True
            logger.info(
                f"Stream ended: No new data for {STREAM_TIMEOUT} seconds. "
                f"Total received: {data_store['total_received']}"
            )
            return True

        return data_store["stream_ended"]


def run_anomaly_detection():
    """
    Background thread running stride-1 sliding window FCVAE detection.

    For each new hourly sample (once 24 hours of history exist):
    - Scores the latest 24-hour window
    - Checks the newest hour against the point threshold → "1-hour anomaly"
    - Checks the full window against k >= 3 criterion → "anomalous day"
    """
    detection_interval = 1  # seconds (check frequency)

    logger.info("Starting FCVAE anomaly detection thread (stride-1)")
    logger.info(f"Detection interval: {detection_interval}s")
    logger.info(f"Window size: {WINDOW_SIZE} hours")
    logger.info(f"Detector config: {detector.get_stats()}")

    while True:
        try:
            check_stream_ended()

            with data_lock:
                data_list = list(data_store["data"])
                samples_since_last = data_store["samples_since_last_window"]

            # Need at least a full window before detection can start
            if len(data_list) < WINDOW_SIZE:
                time.sleep(detection_interval)
                continue

            # Stride 1: process each new sample's window
            if samples_since_last < 1:
                time.sleep(detection_interval)
                continue

            # Process all pending windows (one per new sample)
            num_pending = samples_since_last
            for offset in range(num_pending - 1, -1, -1):
                end_pos = len(data_list) - offset
                start_pos = end_pos - WINDOW_SIZE
                if start_pos < 0:
                    continue

                try:
                    window_data = data_list[start_pos:end_pos]
                    df = pd.DataFrame(window_data)
                    result = detector.score_window_detailed(df)

                    with data_lock:
                        data_store["windows_processed"] += 1

                        if result is None:
                            if data_store["windows_processed"] == 1:
                                logger.warning("score_window_detailed returned None — model may not be loaded")
                            continue

                        newest_ts = str(result["timestamps"][-1])
                        newest_val = result["values"][-1]
                        newest_score = float(result["point_scores"][-1])
                        newest_anomalous = result["is_anomaly"]

                        # Attach NLL score to the data point (shared dict reference)
                        window_data[-1]["nll_score"] = round(newest_score, 4)

                        logger.info(
                            f"  Window scored: {newest_ts} "
                            f"value={newest_val} NLL={newest_score:.4f} "
                            f"threshold={result['point_threshold']:.4f} "
                            f"-> {'ANOMALY' if newest_anomalous else 'normal'}"
                        )

                        # Get ground truth anomaly flag for the newest point
                        newest_is_gt_anomaly = bool(window_data[-1].get("is_anomaly", 0))

                        # 1-hour anomaly check with PA-caught logic
                        if newest_anomalous:
                            # Direct detection → red dot
                            data_store["point_anomalies"].append({
                                "timestamp": newest_ts,
                                "value": newest_val,
                                "point_score": round(newest_score, 4),
                            })
                            data_store["_segment_detected"] = True
                            logger.info(
                                f"  1-HOUR ANOMALY: {newest_ts} "
                                f"value={newest_val} NLL={newest_score:.4f} "
                                f"< threshold={result['point_threshold']:.4f}"
                            )
                        elif data_store["_segment_detected"] and newest_is_gt_anomaly:
                            # PA-caught → orange dot (GT anomaly in detected segment)
                            data_store["pa_caught_anomalies"].append({
                                "timestamp": newest_ts,
                                "value": newest_val,
                                "point_score": round(newest_score, 4),
                            })
                            logger.info(
                                f"  PA-CAUGHT: {newest_ts} "
                                f"value={newest_val} NLL={newest_score:.4f} "
                                f"(ground truth anomaly in detected segment)"
                            )
                        elif not newest_is_gt_anomaly:
                            # Not a ground truth anomaly → segment ends
                            data_store["_segment_detected"] = False

                        # Anomalous hour: last-point decision
                        if result["is_anomaly"]:
                            if not data_store["last_window_anomalous"]:
                                # New anomalous region starts
                                data_store["window_anomalies"].append({
                                    "window_start": str(result["timestamps"][0]),
                                    "window_end": str(result["timestamps"][-1]),
                                    "num_anomalous": result["num_anomalous_points"],
                                    "window_score": round(result["last_point_score"], 4),
                                })
                                logger.info(
                                    f"  ANOMALY DETECTED: last-point at {result['timestamps'][-1]}, "
                                    f"score={result['last_point_score']:.4f}, "
                                    f"mode={result['decision_mode']}"
                                )
                            else:
                                # Extend current anomalous region
                                if data_store["window_anomalies"]:
                                    data_store["window_anomalies"][-1]["window_end"] = str(result["timestamps"][-1])
                                    data_store["window_anomalies"][-1]["num_anomalous"] = max(
                                        data_store["window_anomalies"][-1]["num_anomalous"],
                                        result["num_anomalous_points"],
                                    )
                            data_store["last_window_anomalous"] = True
                        else:
                            data_store["last_window_anomalous"] = False

                        data_store["last_detection"] = pd.Timestamp.now().isoformat()

                except Exception as e:
                    logger.error(f"Error scoring window ending at offset {offset}: {e}", exc_info=True)

            # Always reset regardless of per-window exceptions
            with data_lock:
                data_store["samples_since_last_window"] = max(0, data_store["samples_since_last_window"] - num_pending)

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}", exc_info=True)

        time.sleep(detection_interval)


# ============================================
# Theme colors
# ============================================
PRIMARY_COLOR = "#2c3e50"
SECONDARY_COLOR = "#3498db"
ANOMALY_COLOR = "#e74c3c"
SUCCESS_COLOR = "#27ae60"
BACKGROUND_COLOR = "#ecf0f1"

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "FCVAE Anomaly Detection"

# Health endpoint for producer coordination
@app.server.route("/health")
def health_check():
    """Return health status including Spark readiness."""
    from flask import jsonify
    with data_lock:
        spark_ready = data_store.get("spark_ready", False)
    return jsonify({
        "status": "ready" if spark_ready else "starting",
        "spark_ready": spark_ready,
        "detector": "fcvae",
        "combo": COMBO_STR,
    })

app.layout = dbc.Container([
    # Header
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand(
                "FCVAE Real-Time Anomaly Detection",
                style={"color": "white", "fontSize": "24px", "fontWeight": "bold"}
            ),
        ]),
        color=PRIMARY_COLOR,
        dark=True,
        sticky="top",
        className="mb-4"
    ),

    # Stream ended alert banner (hidden by default)
    dbc.Alert(
        [
            html.H4("Stream Complete", className="alert-heading"),
            html.P(
                "All data has been processed. No more incoming data.",
                className="mb-0"
            ),
        ],
        id="stream-ended-alert",
        color="warning",
        is_open=False,
        className="mb-4"
    ),

    # Status cards row
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Total Received", className="text-center"),
                html.H2(id="total-received", className="text-center text-success")
            ])
        ]), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Hours Scored", className="text-center"),
                html.H2(id="window-count", className="text-center text-info")
            ])
        ]), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("1-Hour Anomalies", className="text-center"),
                html.H2(id="point-anomaly-count", className="text-center text-warning")
            ])
        ]), width=4),
    ], className="mb-4"),

    # Main time series chart with anomalies
    dbc.Card([
        dbc.CardHeader(
            html.H4("Time Series with Anomaly Detection", className="mb-0"),
            style={"backgroundColor": SECONDARY_COLOR, "color": "white"}
        ),
        dbc.CardBody([
            dcc.Graph(id="stream-graph", style={"height": "400px"})
        ])
    ], className="mb-4"),

    # Row 1: Recent records + 1-Hour Anomalies
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H4("Recent Records", className="mb-0"),
                    style={"backgroundColor": PRIMARY_COLOR, "color": "white"}
                ),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="data-table",
                        columns=[
                            {"name": "Timestamp", "id": "timestamp"},
                            {"name": "Value", "id": "value"},
                            {"name": "NLL Score", "id": "nll_score"},
                        ],
                        style_table={"overflowX": "auto", "maxHeight": "300px", "overflowY": "auto"},
                        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "12px"},
                        style_header={"backgroundColor": PRIMARY_COLOR, "color": "white", "fontWeight": "bold"},
                        page_size=10
                    )
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H4("1-Hour Anomalies", className="mb-0"),
                    style={"backgroundColor": "#e67e22", "color": "white"}
                ),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="point-anomaly-table",
                        columns=[
                            {"name": "Timestamp", "id": "timestamp"},
                            {"name": "Value", "id": "value"},
                            {"name": "NLL Score", "id": "point_score"},
                        ],
                        style_table={"overflowX": "auto", "maxHeight": "300px", "overflowY": "auto"},
                        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "12px"},
                        style_header={"backgroundColor": "#e67e22", "color": "white", "fontWeight": "bold"},
                        style_data_conditional=[
                            {
                                "if": {"column_id": "point_score"},
                                "color": "#e67e22",
                                "fontWeight": "bold"
                            }
                        ],
                        page_size=10
                    )
                ])
            ])
        ], width=6),
    ], className="mb-4"),

    # Auto-refresh interval
    dcc.Interval(id="refresh", interval=1000, n_intervals=0),

    # Footer
    html.Footer(
        dbc.Container(
            html.P(
                f"{detector.get_name()} | {WINDOW_SIZE}-Hour Sliding Windows | Combo: {COMBO_STR}",
                className="text-center mb-0",
                style={"color": "white", "padding": "10px"}
            )
        ),
        style={"backgroundColor": PRIMARY_COLOR, "marginTop": "20px"}
    )
], fluid=True, style={"backgroundColor": BACKGROUND_COLOR, "minHeight": "100vh"})


@app.callback(
    [
        Output("stream-graph", "figure"),
        Output("data-table", "data"),
        Output("point-anomaly-table", "data"),
        Output("total-received", "children"),
        Output("window-count", "children"),
        Output("point-anomaly-count", "children"),
        Output("stream-ended-alert", "is_open"),
    ],
    [Input("refresh", "n_intervals")]
)
def update_dashboard(n):
    """Update all dashboard components with latest data."""
    with data_lock:
        data_list = list(data_store["data"])
        point_anomalies = list(data_store["point_anomalies"])
        pa_caught_anomalies = list(data_store["pa_caught_anomalies"])
        total = data_store["total_received"]
        windows_processed = data_store.get("windows_processed", 0)
        stream_ended = data_store.get("stream_ended", False)

    # Create figure
    fig = go.Figure()
    x_range = None

    if data_list:
        # Limit visualization to most recent hours
        max_display_samples = VISUALIZATION_HOURS
        display_data = data_list[-max_display_samples:] if len(data_list) > max_display_samples else data_list
        df = pd.DataFrame(display_data)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Main time series line
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name="Value",
            line=dict(color=SECONDARY_COLOR, width=2),
            hovertemplate="<b>Time:</b> %{x}<br><b>Value:</b> %{y}<extra></extra>"
        ))

        visible_start = df["timestamp"].iloc[0]
        visible_end = df["timestamp"].iloc[-1]
        x_range = [visible_start, visible_end]

        # Circle markers for 1-hour anomalies
        if point_anomalies:
            pa_df = pd.DataFrame(point_anomalies)
            visible_pa = pa_df[
                (pa_df["timestamp"] >= visible_start) &
                (pa_df["timestamp"] <= visible_end)
            ]
            if not visible_pa.empty:
                fig.add_trace(go.Scatter(
                    x=visible_pa["timestamp"],
                    y=visible_pa["value"],
                    mode="markers",
                    name="1-Hour Anomaly",
                    marker=dict(
                        color=ANOMALY_COLOR,
                        size=12,
                        symbol="circle",
                        line=dict(width=2, color="white")
                    ),
                    hovertemplate="<b>1-HR ANOMALY</b><br>Time: %{x}<br>Value: %{y}<extra></extra>"
                ))

        # Orange markers for PA-caught hours (ground truth anomaly in a detected segment)
        if pa_caught_anomalies:
            pc_df = pd.DataFrame(pa_caught_anomalies)
            visible_pc = pc_df[
                (pc_df["timestamp"] >= visible_start) &
                (pc_df["timestamp"] <= visible_end)
            ]
            if not visible_pc.empty:
                fig.add_trace(go.Scatter(
                    x=visible_pc["timestamp"],
                    y=visible_pc["value"],
                    mode="markers",
                    name="Caught (PA)",
                    marker=dict(
                        color="#f39c12",
                        size=12,
                        symbol="circle",
                        line=dict(width=2, color="white")
                    ),
                    hovertemplate="<b>CAUGHT (PA)</b><br>Time: %{x}<br>Value: %{y}<extra></extra>"
                ))

    fig.update_layout(
        xaxis=dict(title="Timestamp", showgrid=True, gridcolor="#e0e0e0", range=x_range),
        yaxis=dict(title="Value", showgrid=True, gridcolor="#e0e0e0"),
        legend=dict(x=0, y=1.1, orientation="h"),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=30, b=50)
    )

    if not data_list:
        fig.add_annotation(
            text="Waiting for data from Kafka...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )

    # Prepare table data
    recent_data = list(reversed(data_list[-10:])) if data_list else []
    point_anomaly_data = list(reversed(point_anomalies[-20:])) if point_anomalies else []

    return (
        fig,
        recent_data,
        point_anomaly_data,
        f"{total:,}",
        f"{windows_processed}",
        f"{len(point_anomalies)}",
        stream_ended,
    )


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("FCVAE Anomaly Detection Dashboard")
    logger.info("=" * 60)
    logger.info(f"Detector: {detector.get_name()}")
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Combo: {COMBO_STR} -> {COMBO}")
    logger.info(f"Window Size: {WINDOW_SIZE} hours")
    logger.info(f"Detector Ready: {detector.is_ready}")
    logger.info(f"Min Samples Required: {detector.min_samples_required}")
    for key, value in detector.get_stats().items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    # Start Spark Streaming in background thread
    streaming_thread = threading.Thread(target=start_spark_streaming, daemon=True)
    streaming_thread.start()

    # Start anomaly detection in background thread
    detection_thread = threading.Thread(target=run_anomaly_detection, daemon=True)
    detection_thread.start()

    # Run Dash app
    logger.info("Starting Dash server on http://0.0.0.0:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
