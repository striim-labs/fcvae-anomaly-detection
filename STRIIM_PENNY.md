# Striim FCVAE Penny Transaction Anomaly Detection Pipeline: Setup Guide

**Striim Version:** Platform 5.2.0.4 (OpenJDK 11)
**Pipeline:** FileReader -> Parse + Filter (amount < $1.00) -> Hourly Aggregation -> 24-Hour Sliding Window -> Open Processor (FCVAE API) -> FileWriter (JSON)

This guide walks through creating and running the standalone penny transaction anomaly detection pipeline in Striim. It assumes:

- Striim Platform is installed and running locally
- The `fcvae` namespace and types already exist from a prior `FCVAE` or `FCVAE_RT` deployment
- The `FCVAEScoreCaller.scm` module is already installed in `$STRIIM_HOME/modules/`
- The FCVAE scoring API is running with the `Penny_All` detector loaded

If you are starting from a completely clean Striim install with no prior FCVAE setup, follow Steps 1-5 of [STRIIM.md](STRIIM.md) first to create the namespace, types, and OP module.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Start the Scoring API](#2-start-the-scoring-api)
3. [Prepare the Data Directory](#3-prepare-the-data-directory)
4. [Verify Existing Types and Load the OP Module](#4-verify-existing-types-and-load-the-op-module)
5. [Create the Penny Application via TQL](#5-create-the-penny-application-via-tql)
6. [Add the Open Processor via Flow Designer](#6-add-the-open-processor-via-flow-designer)
7. [Deploy and Run](#7-deploy-and-run)
8. [Trigger Data Ingestion](#8-trigger-data-ingestion)
9. [Verify Output](#9-verify-output)
10. [Teardown and Cleanup](#10-teardown-and-cleanup)
11. [Re-runs](#11-re-runs)
12. [Troubleshooting](#12-troubleshooting)
13. [Data Flow Summary](#13-data-flow-summary)

---

## 1. Prerequisites

### 1.1 Clone the Repository

This project lives on the `penny-transactions` branch. Clone it directly:

```bash
git clone -b penny-transactions https://github.com/striim-labs/fcvae-anomaly-detection.git
cd fcvae-anomaly-detection
```

**Git LFS is required.** The synthetic transaction CSV files in `data/` are tracked with Git LFS. If you already cloned without LFS installed, pull the real files now:

```bash
git lfs pull
```

Verify LFS files were fetched correctly (should show real file sizes, not tiny pointer files):

```bash
ls -lh data/synthetic_transactions.csv
# Expected: ~50-100 MB, NOT 130 bytes
```

This guide assumes the repo is at `/Users/<your-username>/Documents/Striim/fcvae-anomaly-detection/`. Replace `<your-username>` with your actual macOS username throughout this guide.

### 1.2 Striim Platform

**Striim Platform** installed and running (tested on 5.2.0.4). Verify Striim is accessible at your web UI (e.g., `http://<your-ip>:9080`).

Set `STRIIM_HOME` to your Striim installation directory:

```bash
# Example values:
#   /opt/Striim
#   /Users/<your-username>/Striim
export STRIIM_HOME="/path/to/Striim"
```

### 1.3 Docker

**Docker** and **Docker Compose v2+** are required to run the scoring API (see [Step 2](#2-start-the-scoring-api)).

### 1.4 Prior FCVAE Setup

**Prior FCVAE setup completed.** The `fcvae` namespace and types (`fcvae.ScorerResult`, `fcvae.DailyPayloadStream_Type`) must already exist in Striim from a prior `FCVAE` or `FCVAE_RT` deployment. The `FCVAEScoreCaller.scm` module must be installed in `$STRIIM_HOME/modules/`.

If the module is not already installed, the pre-built JAR is included in the repo. Copy it to Striim's modules directory (Striim expects the `.scm` extension):

```bash
cp striim/fcvae-score-caller/target/FCVAEScoreCaller.jar "$STRIIM_HOME/modules/FCVAEScoreCaller.scm"
```

### 1.5 Model and Data (already in repo)

**Penny model artifacts** are included in the repo at `models/fcvae/Penny_All/` (model.pt, scaler.pkl, scorer.pkl). The oracle threshold is in `models/fcvae/oracle_thresholds.json`. No training is required.

**Synthetic data with penny distribution** is included at `data/synthetic_transactions.csv` (fetched via Git LFS in Step 1.1). The CSV columns are: `timestamp,network_type,transaction_type,amount,is_anomaly,split,penny_is_anomaly`.

---

## 2. Start the Scoring API

The FCVAE scoring API must be running with the `Penny_All` detector loaded before starting the Striim application. No local Python virtual environment is needed if you use Docker.

### Docker 

```bash
cd /Users/<your-username>/Documents/Striim/fcvae-anomaly-detection
docker compose build api
docker compose up -d api
```

The container installs all dependencies automatically. The API will be available at `http://localhost:8000`.

### Verify the API

Regardless of which option you chose, verify `Penny_All` is loaded:

```bash
curl -s http://localhost:8000/v1/model/info | python3 -m json.tool
```

Confirm `Penny_All` appears in the response with threshold `-21.3322`. You should see all 5 detectors loaded: Accel_CMP, Accel_nopin, Star_CMP, Star_nopin, and Penny_All.

---

## 3. Prepare the Data Directory

The penny application uses a separate watched directory from the combo application to keep output files isolated.

```bash
mkdir -p /tmp/fcvae_penny_test
```

**Do not copy the data file in yet.** FileReader tracks files by name and position. If the file already exists in the watched directory when the application starts, it will be treated as "already processed" and skipped. The correct workflow is: start the Striim application first, then copy the file in (see [Step 8](#8-trigger-data-ingestion)).

---

## 4. Verify Existing Types and Load the OP Module

Open the Striim console (web console or CLI via `$STRIIM_HOME/bin/console.sh`).

### 4.1 Verify Types Exist

```sql
USE fcvae;
LIST TYPES;
```

Confirm that `fcvae.ScorerResult` and `fcvae.DailyPayloadStream_Type` appear in the list. These are the two types the Open Processor depends on. They should already exist from a prior FCVAE or FCVAE_RT deployment.

If they are missing, create them:

```sql
USE fcvae;

CREATE TYPE ScorerResult (
  combo_key     String,
  is_anomaly    String,
  anomaly_score String,
  threshold     String,
  window_end    String
);

CREATE TYPE DailyPayloadStream_Type (
  combo_key    String,
  values_list  String,
  window_size  Integer,
  window_start java.lang.String,
  window_end   java.lang.String
);
```

### 4.2 Load the Open Processor Module

OPs do not always persist across Striim restarts. Load it to be safe:

```sql
LOAD OPEN PROCESSOR "/opt/Striim/modules/FCVAEScoreCaller.scm";
```

Replace `/opt/Striim` with your actual `$STRIIM_HOME` path. The Striim console does not expand shell variables.

If the module is already loaded, you will get a no-op confirmation or an "already loaded" message. Both are fine.

---

## 5. Create the Penny Application via TQL

Paste the following into the Striim console. This creates everything except the Open Processor (which must be added via Flow Designer in the next step).

```sql
USE fcvae;

CREATE APPLICATION FCVAE_PENNY;

-- ============================================================
-- Ingest raw transaction rows from CSV
-- ============================================================

CREATE SOURCE PennyTxnFileSource USING FileReader (
  directory: '/tmp/fcvae_penny_test',
  wildcard: 'synthetic_transactions*.csv',
  positionByEOF: false
)
PARSE USING DSVParser (
  columndelimiter: ',',
  header: true
)
OUTPUT TO PennyRawTxnStream;

-- ============================================================
-- Parse timestamp and amount from raw WAEvent
-- ============================================================

CREATE CQ PennyParseTxn
INSERT INTO PennyParsedStream
SELECT
  TO_DATEF(SLEFT(TO_STRING(data[0]), 19),
           'yyyy-MM-dd HH:mm:ss')   AS txn_timestamp,
  TO_DOUBLE(TO_STRING(data[3]))      AS amount
FROM PennyRawTxnStream;

-- ============================================================
-- Filter to penny transactions only (amount < $1.00)
-- All combos are pooled into a single "Penny_All" key
-- ============================================================

CREATE CQ PennyFilter
INSERT INTO PennyFilteredStream
SELECT
  'Penny_All'    AS combo_key,
  txn_timestamp  AS txn_timestamp
FROM PennyParsedStream
WHERE amount < 1.00;

-- ============================================================
-- Hourly aggregation (1-hour jumping window, event time)
-- ============================================================

CREATE JUMPING WINDOW PennyHourlyWindow
OVER PennyFilteredStream
KEEP WITHIN 1 HOUR
ON txn_timestamp
PARTITION BY combo_key;

CREATE CQ PennyHourlyCounts
INSERT INTO PennyHourlyCountStream
SELECT
  w.combo_key                        AS combo_key,
  COUNT(*)                           AS txn_count,
  TO_STRING(FIRST(w.txn_timestamp))  AS hour_start,
  TO_STRING(LAST(w.txn_timestamp))   AS hour_end
FROM PennyHourlyWindow w
GROUP BY w.combo_key;

-- ============================================================
-- 24-hour sliding window
-- ============================================================

CREATE WINDOW PennyDailyWindow
OVER PennyHourlyCountStream
KEEP 24 ROWS
PARTITION BY combo_key;

CREATE STREAM PennyDailyPayloadStream OF fcvae.DailyPayloadStream_Type;

CREATE CQ PennyAssembleDailyPayload
INSERT INTO PennyDailyPayloadStream
SELECT
  w.combo_key                              AS combo_key,
  LIST(TO_STRING(w.txn_count))             AS values_list,
  COUNT(*)                                 AS window_size,
  TO_STRING(FIRST(w.hour_start))           AS window_start,
  TO_STRING(LAST(w.hour_start))            AS window_end
FROM PennyDailyWindow w
GROUP BY w.combo_key
HAVING COUNT(*) >= 24;

-- ============================================================
-- Output stream for scored results (populated by Open Processor)
-- ============================================================

CREATE STREAM PennyScorerResultStream OF fcvae.ScorerResult;

-- ============================================================
-- Write scored results to JSON file
-- ============================================================

CREATE TARGET PennyResultFile USING FileWriter (
  directory: '/tmp/fcvae_penny_test',
  filename: 'penny_scored_output'
)
FORMAT USING JSONFormatter ()
INPUT FROM PennyScorerResultStream;

END APPLICATION FCVAE_PENNY;
```

### What this TQL does

- `PennyParsedStream` and `PennyFilteredStream` and `PennyHourlyCountStream` are auto-created by Striim from the CQ `INSERT INTO` statements with inferred types.
- `PennyDailyPayloadStream` is explicitly created as `fcvae.DailyPayloadStream_Type` because the Open Processor's `@PropertyTemplate` annotation expects this type.
- `PennyScorerResultStream` is explicitly created as `fcvae.ScorerResult` for the same reason.
- `PennyFilter` uses `WHERE amount < 1.00` to select only sub-dollar transactions. All combos are pooled into the single key `'Penny_All'`.
- `HAVING COUNT(*) >= 24` prevents partial windows from reaching the OP during initial fill.
- The `FCVAEScoreCaller` OP reads `combo_key` from the input event and passes it in the HTTP body as `"combo": "Penny_All"`. The API routes this to the penny detector automatically.

---

## 6. Add the Open Processor via Flow Designer

This step cannot be done in TQL (Striim limitation DEV-52892).

1. Log into the Striim web UI at `http://<your-ip>:9080`
2. Navigate to **Apps** and find `fcvae.FCVAE_PENNY`
3. Open it in **Flow Designer**
4. You should see the pipeline from `PennyTxnFileSource` through to `PennyResultFile`, with a gap between `PennyDailyPayloadStream` and `PennyScorerResultStream`
5. Click the **"+"** button or drag an **Open Processor** from the component palette
6. Configure it:
   - **Name:** `PennyOP`
   - **Module:** Select `FCVAEScoreCaller` from the dropdown
   - **Input Stream:** `PennyDailyPayloadStream`
   - **Output Stream:** `PennyScorerResultStream`
7. Click **Save**

### Verify Stream Type Matching

The Open Processor's `@PropertyTemplate` annotation expects:

| Annotation field | Value | Meaning |
|---|---|---|
| `outputType` | `DailyPayloadStream_Type_1_0.class` | What Striim sends TO the processor |
| `inputType` | `ScorerResult_1_0.class` | What the processor sends BACK to Striim |

If you get a type mismatch error on deploy, verify the stream types:

```sql
USE fcvae;
LIST STREAMS IN fcvae.FCVAE_PENNY;
```

`PennyDailyPayloadStream` must be typed as `fcvae.DailyPayloadStream_Type` and `PennyScorerResultStream` must be typed as `fcvae.ScorerResult`. If either shows `Global.WAEvent`, drop and recreate it with the correct type.

---

## 7. Deploy and Run

```sql
USE fcvae;
DEPLOY APPLICATION fcvae.FCVAE_PENNY;
START APPLICATION fcvae.FCVAE_PENNY;
```

Verify:

```sql
LIST APPLICATIONS;
-- Should show: fcvae.FCVAE_PENNY | RUNNING
```

In the Striim web UI, open `fcvae.FCVAE_PENNY` in Flow Designer. All components should show green status with 0 records processed (no data has been ingested yet).

---

## 8. Trigger Data Ingestion

The application must be running (Step 7) before copying the data file in.

```bash
cp /Users/<your-username>/Documents/Striim/fcvae-anomaly-detection/data/synthetic_transactions.csv \
   /tmp/fcvae_penny_test/synthetic_transactions.csv
```

The FileReader will detect the new file and begin ingesting. In the Striim UI you should see Total Input climbing immediately. The penny filter will reduce the throughput significantly (only ~2% of transactions are sub-dollar), and the first scored output will not appear until 24 hourly counts have accumulated (24 hours of event-time data).

---

## 9. Verify Output

### Check Output Files

```bash
ls /tmp/fcvae_penny_test/penny_scored_output*
```

Once files appear:

```bash
head -12 /tmp/fcvae_penny_test/penny_scored_output.00
```

Expected output:

```json
[
 {
  "combo_key":"Penny_All",
  "is_anomaly":"false",
  "anomaly_score":"-0.40744465589523315",
  "threshold":"-21.3322",
  "window_end":"2025/01/06 23:08:54.000"
 },
 ...
]
```

Normal penny windows have scores above `-21.3322` (not anomalous). Windows during penny-rate spike injection periods have scores well below the threshold (anomalous).

### Verify Anomaly Detection

```bash
grep '"is_anomaly":"true"' /tmp/fcvae_penny_test/penny_scored_output.* | head -5
```

You should see matches in the output files corresponding to the test-period penny spike windows (approximately days 51-60 of the dataset).

### Monitor in the Web UI

Open `fcvae.FCVAE_PENNY` in Flow Designer. Each component shows a records/sec counter. Expected throughput pattern:

| Component | Approximate rate | Notes |
|---|---|---|
| PennyTxnFileSource | ~124k records/s | All transactions ingested |
| PennyParseTxn | ~125k records/s | Parsing timestamp and amount |
| PennyFilter | ~3.9k records/s | ~2% of transactions are sub-dollar |
| PennyHourlyWindow | fetching | Accumulating hourly counts |
| PennyHourlyCounts | ~3.9k records/s | One count event per hour boundary |
| PennyDailyWindow | fetching | Accumulating 24 hourly counts |
| PennyAssembleDailyPayload | ~20 records/s | One payload per sliding window step |
| PennyOP | ~20 records/s | HTTP scoring calls to the API |
| PennyResultFile | ~57 records/s | JSON output to disk |

---

## 10. Teardown and Cleanup

### Stop the Striim Application

Striim requires strict lifecycle ordering. Do not skip steps.

```sql
USE fcvae;
STOP APPLICATION fcvae.FCVAE_PENNY;
UNDEPLOY APPLICATION fcvae.FCVAE_PENNY;
DROP APPLICATION fcvae.FCVAE_PENNY CASCADE;
```

### Clean Output and Data Files

```bash
rm -f /tmp/fcvae_penny_test/penny_scored_output*
rm -f /tmp/fcvae_penny_test/synthetic_transactions*.csv
```

### Full Directory Removal (optional)

If you want to remove the penny test directory entirely:

```bash
rm -rf /tmp/fcvae_penny_test
```

### What NOT to Clean

Do not drop the `fcvae` namespace or the shared types (`fcvae.ScorerResult`, `fcvae.DailyPayloadStream_Type`). These are shared with the combo application (`FCVAE` / `FCVAE_RT`) and other FCVAE pipelines. Do not unload the `FCVAEScoreCaller` OP module unless you are also tearing down all other FCVAE applications.

---

## 11. Re-runs

FileReader tracks files by name and position. Once a file has been fully processed, re-copying the same filename is ignored.

### Option A: Undeploy/Redeploy (cleanest)

```sql
USE fcvae;
STOP APPLICATION fcvae.FCVAE_PENNY;
UNDEPLOY APPLICATION fcvae.FCVAE_PENNY;
```

```bash
rm -f /tmp/fcvae_penny_test/penny_scored_output*
rm -f /tmp/fcvae_penny_test/synthetic_transactions*.csv
```

```sql
DEPLOY APPLICATION fcvae.FCVAE_PENNY;
START APPLICATION fcvae.FCVAE_PENNY;
```

Then copy the data file in again:

```bash
cp /Users/<your-username>/Documents/Striim/fcvae-anomaly-detection/data/synthetic_transactions.csv \
   /tmp/fcvae_penny_test/synthetic_transactions.csv
```

### Option B: Use a Unique Filename

The wildcard `synthetic_transactions*.csv` matches any suffix. Copy with an incrementing suffix:

```bash
cp /Users/<your-username>/Documents/Striim/fcvae-anomaly-detection/data/synthetic_transactions.csv \
   /tmp/fcvae_penny_test/synthetic_transactions_run2.csv
```

Increment the suffix for each run (`_run3.csv`, `_run4.csv`, etc.). Output will append to new `penny_scored_output.XX` files.

---

## 12. Troubleshooting

### FileReader Not Picking Up the File

The file was already present in `/tmp/fcvae_penny_test/` when the app started. FileReader considers pre-existing files as processed. Solution: start the app first, then copy the file in.

### No Scored Output After Several Minutes

The 24-hour sliding window needs 24 hourly counts before emitting. The data must span at least 24 complete hour boundaries of event time. Check that the `PennyHourlyCounts` CQ is emitting events in the Striim UI. If `PennyFilter` shows 0 records/s, the data file may not contain sub-dollar transactions (ensure it was generated with `--include-penny`).

### "Stream type mismatch" on Deploy

The stream types do not match the Open Processor's annotation. Verify:

```sql
USE fcvae;
LIST STREAMS IN fcvae.FCVAE_PENNY;
```

`PennyDailyPayloadStream` must be `fcvae.DailyPayloadStream_Type` and `PennyScorerResultStream` must be `fcvae.ScorerResult`. If either is `Global.WAEvent`, drop and recreate with the correct type.

### Empty JSON Output

The output stream is typed as `Global.WAEvent` instead of `fcvae.ScorerResult`. JSONFormatter cannot serialize untyped payloads with named fields. Drop and recreate the stream with the correct type.

### API Returns 503 for Penny_All

The `Penny_All` detector is not loaded in the API. Check that `models/fcvae/Penny_All/` contains `model.pt`, `scaler.pkl`, and `scorer.pkl`, and that `oracle_thresholds.json` contains the `Penny_All` entry. Restart the API to reload.

### ZLIB Input Stream Error

You overwrote a loaded `.scm` without doing a full Striim restart. Follow the module update procedure in [STRIIM.md Step 11](STRIIM.md#11-teardown-and-re-runs).

### Open Processor Property Values Are Null

This is a known issue. The `@PropertyTemplateProperty` annotation values do not reliably inject at runtime. The Java code has fallback defaults: the API endpoint defaults to `http://localhost:8000/v1/score` and the timeout defaults to 5000ms. For demo purposes this works without changes.

---

## 13. Data Flow Summary

```
synthetic_transactions.csv (all combos, includes penny-amount transactions)
    |
    v
PennyTxnFileSource (FileReader + DSVParser, watches /tmp/fcvae_penny_test/)
    |
    v
PennyRawTxnStream (WAEvent: data[0]=timestamp, data[3]=amount, ...)
    |
    v
PennyParseTxn CQ (parses txn_timestamp as DateTime, amount as Double)
    |
    v
PennyParsedStream (txn_timestamp, amount)
    |
    v
PennyFilter CQ (WHERE amount < 1.00, sets combo_key = 'Penny_All')
    |
    v
PennyFilteredStream (combo_key='Penny_All', txn_timestamp)  [~2% of input]
    |
    v
PennyHourlyWindow (1-hour jumping, ON txn_timestamp, PARTITION BY combo_key)
    |
    v
PennyHourlyCounts CQ (COUNT per hour)
    |
    v
PennyHourlyCountStream (combo_key, txn_count, hour_start, hour_end)
    |
    v
PennyDailyWindow (KEEP 24 ROWS, PARTITION BY combo_key)
    |
    v
PennyAssembleDailyPayload CQ (LIST() of 24 counts, HAVING COUNT >= 24)
    |
    v
PennyDailyPayloadStream OF fcvae.DailyPayloadStream_Type
    |
    v
PennyOP (FCVAEScoreCaller -- HTTP POST to localhost:8000/v1/score with combo=Penny_All)
    |
    v
PennyScorerResultStream OF fcvae.ScorerResult
    |
    v
PennyResultFile (FileWriter + JSONFormatter -> /tmp/fcvae_penny_test/penny_scored_output)
```

---

## Environment Reference

| Component | Detail |
|---|---|
| Striim Platform | 5.2.0.4, installed at `$STRIIM_HOME` |
| FCVAE API | `http://localhost:8000` |
| Striim namespace | `fcvae` |
| Application name | `fcvae.FCVAE_PENNY` |
| OP module name | `FCVAEScoreCaller` (shared with combo application) |
| OP instance name | `PennyOP` |
| Penny model | `models/fcvae/Penny_All/` (model.pt, scaler.pkl, scorer.pkl) |
| Oracle threshold | `-21.3322` (in `models/fcvae/oracle_thresholds.json`) |
| Data file | `<repo>/data/synthetic_transactions.csv` (copy to `/tmp/fcvae_penny_test/` at runtime) |
| Output directory | `/tmp/fcvae_penny_test/` |
| Output files | `penny_scored_output.00`, `.01`, `.02`, ... |
| Striim logs | `$STRIIM_HOME/logs/striim.server.log` |