# Striim FCVAE Anomaly Detection Pipeline: Setup Guide

**Striim Version:** Platform 5.2.0.4 (OpenJDK 11)
**Pipeline:** FileReader -> Hourly Aggregation -> 24-Hour Sliding Window -> Open Processor (FCVAE API) -> FileWriter (JSON)

This guide walks through recreating the end-to-end Striim FCVAE anomaly detection pipeline from scratch on any local machine. It assumes Striim Platform is installed at `/opt/Striim` and the FCVAE scoring API is running locally.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Start the FCVAE Scoring API](#2-start-the-fcvae-scoring-api)
3. [Prepare the Data File](#3-prepare-the-data-file)
4. [Install the Open Processor Module](#4-install-the-open-processor-module)
5. [Create the Namespace and Types in Striim](#5-create-the-namespace-and-types-in-striim)
6. [Import the TQL Application](#6-import-the-tql-application)
7. [Add the Open Processor via Flow Designer](#7-add-the-open-processor-via-flow-designer)
8. [Deploy and Run](#8-deploy-and-run)
9. [Trigger Data Ingestion](#9-trigger-data-ingestion)
10. [Verify Output](#10-verify-output)
11. [Teardown and Re-runs](#11-teardown-and-re-runs)
12. [Troubleshooting](#12-troubleshooting)
13. [Full TQL Reference](#13-full-tql-reference)
14. [Building from Source (Developer Reference)](#14-building-from-source-developer-reference)

---

## 1. Prerequisites

Before starting, confirm you have the following:

**Striim Platform** installed and running (tested on 5.2.0.4). The default install path is `/opt/Striim`. Verify Striim is accessible at your web UI (e.g., `http://<your-ip>:9080`).

**The FCVAE repo** cloned locally. This guide assumes the repo is at:
```
/Users/<your-username>/Documents/Striim/fcvae-anomaly-detection/
```
Replace `<your-username>` with your actual macOS username throughout this guide.

**Python 3.11+** and **UV** package manager for the scoring API.

**Docker** (optional, if you prefer to run the scoring API via Docker instead of locally).

---

## 2. Start the FCVAE Scoring API

The Striim pipeline calls the FCVAE scoring API over HTTP at `http://localhost:8000/v1/score`. The API must be running before starting the Striim application.

### Option A: Local (recommended for demo)

```bash
cd /Users/<your-username>/Documents/Striim/fcvae-anomaly-detection

# Install API dependencies
cd api && uv sync && cd ..

# Start the API server
api/.venv/bin/uvicorn api.main:app --port 8000
```

### Option B: Docker

```bash
cd /Users/<your-username>/Documents/Striim/fcvae-anomaly-detection
docker compose build api
docker compose up api
```

### Verify the API is healthy

```bash
curl http://localhost:8000/health
```

You should see a JSON response with `"status": "healthy"` and all four combos loaded (Accel_CMP, Accel_nopin, Star_CMP, Star_nopin).

---

## 3. Prepare the Data File

The synthetic transaction data file lives inside the repo at:

```
/Users/<your-username>/Documents/Striim/fcvae-anomaly-detection/data/synthetic_transactions_phase2.csv
```

Striim's FileReader is configured to watch `/tmp/fcvae_test/` for files matching `synthetic_transactions_phase2.csv`. Create this directory if it does not already exist:

```bash
mkdir -p /tmp/fcvae_test
```

**Do not copy the data file in yet.** FileReader tracks files by name and position. If the file already exists in the watched directory when the application starts, it will be treated as "already processed" and skipped. The correct workflow is: start the Striim application first, then copy the file in (see [Step 9](#9-trigger-data-ingestion)).

---

## 4. Install the Open Processor Module

The Open Processor is a pre-built Java module that Striim loads at runtime. It calls the FCVAE scoring API over HTTP for each event batch and returns typed scoring results.

The pre-built module is included in the repo at `striim/fcvae-score-caller/target/FCVAEScoreCaller.jar`. Copy it to Striim's modules directory with the `.scm` extension:

```bash
cp /path/to/fcvae-anomaly-detection/striim/fcvae-score-caller/target/FCVAEScoreCaller.jar \
   /opt/Striim/modules/FCVAEScoreCaller.scm
```

Then restart Striim so it picks up the new module

> **Note:** If you need to modify the Open Processor Java code and rebuild from source, see [Step 14](#14-building-from-source-developer-reference).

---

## 5. Create the Namespace and Types in Striim

Open the Striim console. You can use either the CLI console (`/opt/Striim/bin/console.sh`) or paste commands into the web console. The web console does NOT support `@/path/to/file.tql` syntax, so paste directly.

**Always use the `fcvae` namespace.** Components created in the wrong namespace cause deployment failures.

### 5.1 Create the Namespace

```sql
CREATE NAMESPACE fcvae;
USE fcvae;
```

### 5.2 Create Custom Types

These types must exist before the TQL application can reference them.

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

Use `LIST TYPES;` to verify type names. Striim registers types as `fcvae.ScorerResult` but generates Java classes as `wa.fcvae.ScorerResult_1_0`. The stream declarations use the short name (`fcvae.ScorerResult`), while the Java `@PropertyTemplate` annotation uses the generated class name (`ScorerResult_1_0`).

---

## 6. Load Open Processor and Import the TQL Application

```sql
LOAD OPEN PROCESSOR "/opt/Striim/modules/FCVAEScoreCaller.scm";
```

The TQL defines everything except the Open Processor (which can only be added via Flow Designer). Paste the following into the Striim console. This TQL file is also located at `striim/FCVAE.tql`.

```sql
-- ============================================================================
-- FCVAE: End-to-End Anomaly Detection Pipeline
--
-- This TQL creates everything EXCEPT the Open Processor.
-- After importing, add the Open Processor via Flow Designer:
--   1. Open FCVAE in Flow Designer
--   2. Drag "Open Processor" from Base Components into the workspace
--   3. Set Input Stream: DailyPayloadStream
--   4. Set Output Stream: ScorerResultStream
--   5. Set Processor: FCVAEScoreCaller
--   6. Set properties: apiEndpoint, apiKey, timeoutMs, maxRetries
--   7. Save
--
-- Prerequisites:
--   1. FCVAE Scoring API running at http://localhost:8000
--   2. FCVAEScoreCaller.scm loaded via:
--        LOAD OPEN PROCESSOR '/opt/Striim/modules/FCVAEScoreCaller.scm';
--   3. Synthetic transactions CSV in /tmp/fcvae_test/
--
-- Namespace: fcvae
-- ============================================================================

CREATE APPLICATION FCVAE;

-- ============================================================================
-- Ingest raw transaction rows from CSV
-- ============================================================================

CREATE SOURCE TxnFileSource USING FileReader (
  directory: '/tmp/fcvae_test',
  wildcard: 'synthetic_transactions_phase2.csv',
  positionByEOF: false
)
PARSE USING DSVParser (
  columndelimiter: ',',
  header: true
)
OUTPUT TO RawTxnStream;

-- ============================================================================
-- Parse raw WAEvent into typed fields
-- ============================================================================

CREATE CQ ParseTransactions
INSERT INTO TypedTxnStream
SELECT
  TO_STRING(data[1]) + '_' + CASE
    WHEN TO_STRING(data[2]) = 'no-pin' THEN 'nopin'
    ELSE TO_STRING(data[2])
  END                                               AS combo_key,
  TO_DATEF(SLEFT(TO_STRING(data[0]), 19),
           'yyyy-MM-dd HH:mm:ss')                   AS txn_timestamp
FROM RawTxnStream;

-- ============================================================================
-- Aggregate into hourly counts per combo
-- ============================================================================

CREATE JUMPING WINDOW HourlyWindow
OVER TypedTxnStream
KEEP WITHIN 1 HOUR
ON txn_timestamp
PARTITION BY combo_key;

CREATE CQ HourlyCounts
INSERT INTO HourlyCountStream
SELECT
  w.combo_key                        AS combo_key,
  COUNT(*)                           AS txn_count,
  TO_STRING(FIRST(w.txn_timestamp))  AS hour_start,
  TO_STRING(LAST(w.txn_timestamp))   AS hour_end
FROM HourlyWindow w
GROUP BY w.combo_key;

-- ============================================================================
-- Assemble 24-hour sliding window payloads
-- ============================================================================

CREATE WINDOW DailyWindow
OVER HourlyCountStream
KEEP 24 ROWS
PARTITION BY combo_key;

CREATE CQ AssembleDailyPayload
INSERT INTO DailyPayloadStream
SELECT
  w.combo_key                              AS combo_key,
  LIST(TO_STRING(w.txn_count))             AS values_list,
  COUNT(*)                                 AS window_size,
  TO_STRING(FIRST(w.hour_start))           AS window_start,
  TO_STRING(LAST(w.hour_start))            AS window_end
FROM DailyWindow w
GROUP BY w.combo_key
HAVING COUNT(*) >= 24;

-- ============================================================================
-- Output stream for scored results (populated by Open Processor)
--
-- The Open Processor (added via Flow Designer) reads DailyPayloadStream
-- and writes ScorerResult_1_0 events to ScorerResultStream.
-- ============================================================================

CREATE STREAM ScorerResultStream OF fcvae.ScorerResult;

-- ============================================================================
-- Write scored results to JSON file for validation
-- ============================================================================

CREATE TARGET ResultFile USING FileWriter (
  directory: '/tmp/fcvae_test',
  filename: 'scored_output'
)
FORMAT USING JSONFormatter ()
INPUT FROM ScorerResultStream;

END APPLICATION FCVAE;
```

**What this does not include:** The Open Processor. Striim does not support `CREATE OPENPROCESSOR` in TQL (this is a known limitation, DEV-52892). You must add it via the Flow Designer UI in the next step.

---

## 7. Add the Open Processor via Flow Designer

This is the critical step that cannot be done in TQL.

### 7.1 Open the Application in Flow Designer

1. Log into the Striim web UI
2. Navigate to **Apps** and find `fcvae.FCVAE`
3. Click the application to open it in **Flow Designer**

You should see the pipeline components laid out: `TxnFileSource` -> streams -> CQs -> `DailyPayloadStream` and `ScorerResultStream` -> `ResultFile`. There will be a gap between `DailyPayloadStream` and `ScorerResultStream` where the Open Processor needs to go.

### 7.2 Add the Open Processor

1. In the Flow Designer canvas, click the **"+"** button or drag an **Open Processor** component from the component palette
2. Configure it:
   - **Name:** `OP` (or any name you prefer)
   - **Module:** Select `FCVAEScoreCaller` from the dropdown (this is the `.scm` module you loaded in Step 4)
   - **Input Stream:** `DailyPayloadStream`
   - **Output Stream:** `ScorerResultStream`
3. Set the Open Processor properties (these may or may not inject at runtime due to the known property injection issue, but set them anyway):
   - `apiEndpoint`: `http://localhost:8000/v1/score`
   - `apiKey`: (leave empty)
   - `timeoutMs`: `5000`

### 7.3 Verify Stream Type Matching

The Open Processor's `@PropertyTemplate` annotation declares:

| Annotation field | Value | Meaning |
|---|---|---|
| `outputType` | `DailyPayloadStream_Type_1_0.class` | What Striim sends TO the processor (the input events) |
| `inputType` | `ScorerResult_1_0.class` | What the processor sends BACK to Striim (the output events) |

**Important naming convention:** From Striim's perspective, `outputType` means "what I output to the processor" and `inputType` means "what I receive as input from the processor." This is the reverse of how a Java developer would think about it.

The input stream (`DailyPayloadStream`) must be typed as `fcvae.DailyPayloadStream_Type`, and the output stream (`ScorerResultStream`) must be typed as `fcvae.ScorerResult`. If either is `Global.WAEvent`, you will get a type mismatch error on deploy. To fix, drop and recreate the stream with the correct type:

```sql
USE fcvae;
DROP STREAM ScorerResultStream;
CREATE STREAM ScorerResultStream OF fcvae.ScorerResult;
```

### 7.4 Save the Application

Click **Save** in the Flow Designer. The application should now show all components connected.

---

## 8. Deploy and Run

```sql
USE fcvae;
DEPLOY APPLICATION fcvae.FCVAE;
START APPLICATION fcvae.FCVAE;
```

Verify the application status:

```sql
LIST APPLICATIONS;
-- Should show: fcvae.FCVAE | RUNNING
```

---

## 9. Trigger Data Ingestion

Because FileReader tracks file positions, the data file must appear "new" to Striim. The app should already be running (Step 8) before you copy the file in.

### First Run

Copy the file from the repo into the watched directory:

```bash
cp /Users/<your-username>/Documents/Striim/fcvae-anomaly-detection/data/synthetic_transactions_phase2.csv \
   /tmp/fcvae_test/synthetic_transactions_phase2.csv
```

### Subsequent Re-runs

FileReader ignores files it has already processed (tracked by filename). For re-runs, copy with a unique filename that still matches the wildcard pattern:

```bash
cp /Users/<your-username>/Documents/Striim/fcvae-anomaly-detection/data/synthetic_transactions_phase2.csv \
   /tmp/fcvae_test/synthetic_transactions_phase2_run2.csv
```

Increment the suffix for each run (`_run3.csv`, `_run4.csv`, etc.).

---

## 10. Verify Output

### Check Striim Logs

```bash
tail -f /opt/Striim/logs/striim.server.log
```

Look for log entries from `FCVAEScoreCaller` showing successful API calls.

### Check the Output File

```bash
cat /tmp/fcvae_test/scored_output.00
```

You should see JSON entries like:

```json
{
  "combo_key": "Star_CMP",
  "is_anomaly": "false",
  "anomaly_score": "-0.31990835070610046",
  "threshold": "-1.756",
  "window_end": "2025/02/25 23:01:02.000"
}
```

All four combos (Accel_CMP, Accel_nopin, Star_CMP, Star_nopin) should appear. With normal synthetic data, scores will be above their respective thresholds (not anomalous).

### Monitor in the Web UI

Open the application in Flow Designer. Each component shows a records/sec counter. You should see data flowing from the source through the CQs to the target.

---

## 11. Teardown and Re-runs

### Stop and Clean Up

Striim requires strict lifecycle ordering:

```sql
STOP APPLICATION fcvae.FCVAE;
UNDEPLOY APPLICATION fcvae.FCVAE;
DROP APPLICATION fcvae.FCVAE CASCADE;
```

**Do not skip steps.** Attempting to drop without stopping/undeploying first will fail.

### Module Updates

If you need to rebuild the Open Processor Java code:

```sql
-- In Striim console:
UNLOAD OPEN PROCESSOR "/opt/Striim/modules/FCVAEScoreCaller.scm";
```

Then:

```bash
# Stop Striim
/opt/Striim/bin/server.sh stop

# Rebuild and copy (see Step 14)
cd striim/fcvae-score-caller
./build.sh

# Start Striim
/opt/Striim/bin/server.sh start
```

Then reload:

```sql
LOAD OPEN PROCESSOR "/opt/Striim/modules/FCVAEScoreCaller.scm";
```

**Never** overwrite a loaded `.scm` without doing a full Striim restart in between. Doing so causes `Unexpected end of ZLIB input stream` errors.

### Re-running with the Same Data

Use a new filename for each run (the wildcard `synthetic_transactions_phase2*.csv` will match any suffix):

```bash
cp /Users/<your-username>/Documents/Striim/fcvae-anomaly-detection/data/synthetic_transactions_phase2.csv \
   /tmp/fcvae_test/synthetic_transactions_phase2_run3.csv
```

---

## 12. Troubleshooting

### FileReader Not Picking Up the File

The file was already present in `/tmp/fcvae_test/` when the app started. FileReader considers pre-existing files as processed. Solution: start the app first, then copy the file into `/tmp/fcvae_test/`.

### Jumping Window Never Fires

The input data must span at least one complete hour boundary. If your data only covers a partial hour, the window will not emit. Ensure `synthetic_transactions_phase2.csv` contains 27+ hours of data.

### "Stream type mismatch" on Deploy

The stream types do not match the Open Processor's annotation. Check with:

```sql
LIST TYPES;
LIST STREAMS IN fcvae.FCVAE;
```

Ensure `DailyPayloadStream` is typed as `fcvae.DailyPayloadStream_Type` (not `Global.WAEvent`) and `ScorerResultStream` is typed as `fcvae.ScorerResult`.

### ZLIB Input Stream Error

You overwrote a loaded `.scm` without doing a full Striim restart. Follow the module update procedure in Step 11.

### Empty JSON Output

The output stream is typed as `Global.WAEvent` instead of `fcvae.ScorerResult`. JSONFormatter cannot serialize untyped payloads with named fields. Drop and recreate the stream with the correct type.

### Open Processor Property Values Are Null

This is a known issue. The `@PropertyTemplateProperty` annotation values do not reliably inject at runtime. The Java code has fallback defaults: the API endpoint defaults to `http://localhost:8000/v1/score` and the timeout defaults to 5000ms. For demo purposes this works without changes.

### CSV Header Crashes the App

Ensure `header: Yes` is set in the DSVParser configuration. Without it, the parser tries to process the header row as data, and `SLEFT(TO_STRING(data[0]), 19)` on the string `"timestamp"` throws a `StringIndexOutOfBoundsException`.

### Components Created in Wrong Namespace

Always run `USE fcvae;` before creating any components. If you accidentally created components under `admin`, drop them and recreate under `fcvae`.

### "CREATE OPENPROCESSOR" Error

There is no such TQL keyword. Open Processors can only be added through the Flow Designer UI. This is documented as a known Striim limitation (DEV-52892).

---

## 13. Full TQL Reference

### Application Lifecycle Commands

```sql
-- Create namespace (once)
CREATE NAMESPACE fcvae;
USE fcvae;

-- Deploy and run
DEPLOY APPLICATION fcvae.FCVAE;
START APPLICATION fcvae.FCVAE;

-- Inspect
LIST APPLICATIONS;
LIST TYPES;
LIST STREAMS IN fcvae.FCVAE;

-- Tear down
STOP APPLICATION fcvae.FCVAE;
UNDEPLOY APPLICATION fcvae.FCVAE;
DROP APPLICATION fcvae.FCVAE CASCADE;

-- Module management (requires full Striim restart between unload and load)
UNLOAD OPEN PROCESSOR "/opt/Striim/modules/FCVAEScoreCaller.scm";
LOAD OPEN PROCESSOR "/opt/Striim/modules/FCVAEScoreCaller.scm";

-- Export types for Maven build (only needed if rebuilding from source)
EXPORT TYPES OF fcvae.FCVAE TO "UploadedFiles/fcvae_types.jar";
```

### Data Flow Summary

```
TxnFileSource (FileReader + DSVParser, watches /tmp/fcvae_test/)
    |
    v
RawTxnStream (WAEvent: data[0]=timestamp, data[1]=network, data[2]=txn_type, ...)
    |
    v
ParseTransactions CQ (builds combo_key, parses timestamp)
    |
    v
TypedTxnStream (combo_key, txn_timestamp)
    |
    v
HourlyWindow (1-hour jumping, PARTITION BY combo_key, ON txn_timestamp)
    |
    v
HourlyCounts CQ (COUNT per combo per hour, with hour_start and hour_end)
    |
    v
HourlyCountStream (combo_key, txn_count, hour_start, hour_end)
    |
    v
DailyWindow (KEEP 24 ROWS, PARTITION BY combo_key)
    |
    v
AssembleDailyPayload CQ (LIST() of 24 counts, HAVING COUNT >= 24)
    |
    v
DailyPayloadStream (combo_key, values_list, window_size, window_start, window_end)
    |
    v
Open Processor: OP (FCVAEScoreCaller -- HTTP POST to localhost:8000/v1/score)
    |
    v
ScorerResultStream OF fcvae.ScorerResult (combo_key, is_anomaly, anomaly_score, threshold, window_end)
    |
    v
ResultFile (FileWriter + JSONFormatter -> /tmp/fcvae_test/scored_output)
```

### Key Striim Lessons

| Topic | Detail |
|---|---|
| Open Processors are UI-only | Cannot be defined in TQL; must be added via Flow Designer after importing the rest of the application |
| `outputType` / `inputType` naming | From Striim's perspective: `outputType` = what Striim sends TO the OP (the input); `inputType` = what the OP sends BACK to Striim (the output) |
| Dual WAEvent classes | `com.webaction.runtime.containers.WAEvent` (SDK, compile-time) vs `com.webaction.proc.events.WAEvent` (runtime). The Java code uses reflection-based class name checks to handle both |
| `LIST()` ordering | Preserves chronological insertion order in sliding windows |
| FileReader position tracking | Tracks files by name. Re-copying the same filename is ignored. Use wildcards and unique filenames for re-runs |
| Application lifecycle | Strict order: STOP -> UNDEPLOY -> DROP CASCADE. Cannot skip steps |
| ZLIB corruption | Always do a full Striim restart when updating `.scm` files |
| Web console limitations | Does not support `@` file imports. Use CLI console or paste directly |
| `HAVING` in window CQs | Works for filtering by aggregate conditions (e.g., `COUNT(*) >= 24`) |
| Component namespace | Always `USE <namespace>;` before creating components |

---

## 14. Building from Source (Developer Reference)

This section is only needed if you want to modify the Open Processor Java code. Most users can skip this and use the pre-built module from Step 4.

### Repo Structure

```
striim/
├── FCVAE.tql                         # Application TQL
├── lib/
│   └── fcvae_types.jar               # Pre-built Striim types (committed to repo)
└── fcvae-score-caller/
    ├── build.sh                      # One-command build + install script
    ├── pom.xml                       # Maven config (shade plugin produces fat JAR)
    ├── target/
    │   └── FCVAEScoreCaller.jar      # Pre-built module (committed to repo)
    └── src/
        └── main/java/com/striim/fcvae/
            └── FCVAEScoreCaller.java
```

### Prerequisites for Building

**Java:** OpenJDK 11 is required. On macOS with Homebrew: `/opt/homebrew/opt/openjdk@11`. Ensure `JAVA_HOME` is set appropriately.

**Maven** 3.9+ for compiling the Open Processor Java module.

### Install the Types JAR into Maven

The Open Processor Java code imports Striim-exported types (`ScorerResult_1_0`, `DailyPayloadStream_Type_1_0`). A pre-built types JAR containing these classes is included in the repo at `striim/lib/fcvae_types.jar`. Install it into your local Maven repo before building:

```bash
TYPES_JAR="$(pwd)/striim/lib/fcvae_types.jar"

mvn install:install-file \
  -DgroupId=com.striim.fcvae \
  -DartifactId=fcvae-types \
  -Dversion=1.0.0-SNAPSHOT \
  -Dpackaging=jar \
  -Dfile="$TYPES_JAR" \
  -DgeneratePom=true

mvn install:install-file \
  -DgroupId=com.striim.fcvae \
  -DartifactId=FCVAETypes \
  -Dversion=1.0.0-SNAPSHOT \
  -Dpackaging=jar \
  -Dfile="$TYPES_JAR" \
  -DgeneratePom=true
```

The `pom.xml` declares the types dependency under two artifact IDs, so both installs are required.

### Build with build.sh

```bash
cd striim/fcvae-score-caller
chmod +x build.sh
./build.sh
```

What `build.sh` does:

1. **Installs the Striim SDK JAR** from `/opt/Striim/StriimSDK/StriimOpenProcessor-SDK.jar` into your local Maven repo (groupId `com.striim`, artifactId `OpenProcessorSDK`). Safe to re-run.

2. **Runs `mvn clean package`**, which compiles `FCVAEScoreCaller.java` and uses the Maven Shade plugin to produce a fat JAR at `target/FCVAEScoreCaller.jar`. The fat JAR bundles Gson (for JSON parsing) but excludes Striim SDK and types JARs (marked `provided` in the pom since Striim supplies them at runtime). The shade plugin also injects manifest entries that tell Striim the module name, service interface, and implementation class.

3. **Copies `target/FCVAEScoreCaller.jar` to `/opt/Striim/modules/FCVAEScoreCaller.scm`**. Striim expects the `.scm` extension for Open Processor modules. The file is just a renamed JAR.

After a successful build, `target/` will contain:

```
target/
├── classes/com/striim/fcvae/FCVAEScoreCaller.class
├── fcvae-score-caller-1.0-SNAPSHOT.jar   # unshaded JAR (not used)
└── FCVAEScoreCaller.jar                  # shaded fat JAR -> becomes .scm
```

### Restarting Local Steps From Scratch

```sql

USE fcvae;

STOP APPLICATION fcvae.FCVAE;

UNDEPLOY APPLICATION fcvae.FCVAE;

DROP APPLICATION fcvae.FCVAE CASCADE;

```

``` bash

rm -f /tmp/fcvae_test/phase3_scored_output*

rm -f /tmp/fcvae_test/scored_output*

rm -f /tmp/fcvae_test/synthetic_transactions_phase2*.csv

```

```sql

DROP TYPE fcvae.ScorerResult; DROP TYPE fcvae.DailyPayloadStream_Type;

UNLOAD OPEN PROCESSOR "/opt/Striim/modules/FCVAEScoreCaller.scm";

```

---

## Environment Reference

| Component | Detail |
|---|---|
| Striim Platform | 5.2.0.4, installed at `/opt/Striim` |
| FCVAE API | `http://localhost:8000` |
| Striim namespace | `fcvae` |
| OP module name | `FCVAEScoreCaller` |
| Data file | `<repo>/data/synthetic_transactions_phase2.csv` (copy to `/tmp/fcvae_test/` at runtime) |
| Output directory | `/tmp/fcvae_test/` |
| Striim logs | `/opt/Striim/logs/striim.server.log` |
| Types JAR | `<repo>/striim/lib/fcvae_types.jar` (only needed for building from source) |