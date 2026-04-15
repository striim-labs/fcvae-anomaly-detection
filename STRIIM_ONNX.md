# Striim FCVAE Anomaly Detection Pipeline: ONNX Setup Guide

**Striim Version:** Platform 5.2.0.4 (OpenJDK 11)
**Namespace:** `fcvae_onnx` (separate from the HTTP sidecar's `fcvae` namespace)
**Pipeline:** FileReader -> Parse CQ -> Hourly Jumping Window -> 24-Hour Sliding Window -> Open Processor (ONNX Runtime) -> FileWriter (JSON)

This guide walks through setting up the end-to-end Striim FCVAE anomaly detection pipeline using the **ONNX Runtime scorer**. Unlike the HTTP sidecar approach (`STRIIM.md`), this version runs inference in-process via ONNX Runtime's Java JNI bindings. No Python dependency is required at runtime.

This deployment uses the `fcvae_onnx` namespace, which is independent of the `fcvae` namespace used by the HTTP sidecar OP. Both can coexist on the same Striim instance.

This uses the **typed-stream approach** (not the WAEvent pass-through pattern) because the pipeline requires Striim's native windowing and aggregation CQs upstream of the Open Processor. This requires a types JAR, `CREATE TYPE` statements, and the JAR removal trick for deployment.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Deploy to Striim](#2-deploy-to-striim)
3. [Wire the Open Processor in Flow Designer](#3-wire-the-open-processor-in-flow-designer)
4. [Run and Verify](#4-run-and-verify)
5. [Results](#5-results)
6. [Teardown and Re-runs](#6-teardown-and-re-runs)

---

## 1. Prerequisites

| Requirement | Detail |
|---|---|
| Striim Platform | 5.2.0.4, installed at `$STRIIM_HOME` (e.g. `/opt/Striim`) |
| Java | OpenJDK 11 |

**No Python or scoring API required.** The ONNX Runtime scorer runs entirely in-process.

Set `STRIIM_HOME`:

```bash
export STRIIM_HOME="/opt/Striim"
```

Pre-built artifacts in this repo:

| Artifact | Path | Purpose |
|---|---|---|
| OP module (.scm) | `striim/fcvae-onnx-scorer/target/FCVAEOnnxScorer.jar` | Striim Open Processor (fat JAR, ONNX Runtime + Gson shaded in) |
| Types JAR | `striim/fcvae-onnx-scorer/target/fcvae_onnx-types.jar` | Type classes for `$STRIIM_HOME/lib/` (`wa.fcvae_onnx` package) |
| TQL | `striim/FCVAE.tql` | Striim application definition (base -- adjust namespace references) |
| Models | `models/fcvae/{Penny_All,Accel_CMP,Accel_nopin,Star_CMP,Star_nopin}/` | ONNX model files + configs |
| Data | `data/synthetic_transactions_phase2.csv` | Synthetic transaction data (~27.5 hours, 4 combos) |

The Open Processor and types JAR are pre-built. You do not need to compile anything.

---

## 2. Deploy to Striim

This pipeline uses typed streams, which requires the **JAR removal trick**: the types JAR must be absent from `lib/` when creating types (to avoid "class already exists" errors), then restored before loading the OP.

### 2.1 Copy model files

```bash
mkdir -p /opt/Striim/fcvae-models
cp -r models/fcvae/Penny_All /opt/Striim/fcvae-models/
cp -r models/fcvae/Accel_CMP /opt/Striim/fcvae-models/
cp -r models/fcvae/Accel_nopin /opt/Striim/fcvae-models/
cp -r models/fcvae/Star_CMP /opt/Striim/fcvae-models/
cp -r models/fcvae/Star_nopin /opt/Striim/fcvae-models/
```

Each subdirectory must contain `model.onnx`, `model.onnx.data`, and `model_config.json`.

### 2.2 Install artifacts and stop Striim

```bash
cp striim/fcvae-onnx-scorer/target/FCVAEOnnxScorer.jar $STRIIM_HOME/modules/FCVAEOnnxScorer.scm
cp striim/fcvae-onnx-scorer/target/fcvae_onnx-types.jar $STRIIM_HOME/lib/fcvae_onnx-types.jar
```

If Striim is running, stop it with Ctrl+C.

### 2.3 JAR removal trick

```bash
mv $STRIIM_HOME/lib/fcvae_onnx-types.jar /tmp/fcvae_onnx-types.jar
$STRIIM_HOME/bin/server.sh
```

### 2.4 Create namespace, types, and import TQL

In the Striim console:

```sql
CREATE NAMESPACE fcvae_onnx;
USE fcvae_onnx;

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

Then paste the contents of `striim/FCVAE_ONNX.tql` into the console (everything from `CREATE APPLICATION FCVAE` through `END APPLICATION FCVAE`).

### 2.5 Restore types JAR, restart, and load OP

Stop Striim with Ctrl+C, then:

```bash
mv /tmp/fcvae_onnx-types.jar $STRIIM_HOME/lib/fcvae_onnx-types.jar
$STRIIM_HOME/bin/server.sh
```

In the Striim console:

```sql
USE fcvae_onnx;
LOAD OPEN PROCESSOR "/opt/Striim/modules/FCVAEOnnxScorer.scm";
```

Verify:

```sql
LIST OPENPROCESSORS;
```

---

## 3. Wire the Open Processor in Flow Designer

1. In the Striim web UI, open `fcvae_onnx.FCVAE` in **Flow Designer**
2. Drag **Open Processor** from Base Components into the workspace
3. Configure:
   - **Module:** `FCVAEOnnxScorer`
   - **Input Stream:** `DailyPayloadStream`
   - **Output Stream:** `ScorerResultStream`
   - **modelsDir:** `/opt/Striim/fcvae-models` (default, change if models are elsewhere)
4. Click **Save**

---

## 4. Run and Verify

### 4.1 Deploy and start

```sql
USE fcvae_onnx;
DEPLOY APPLICATION fcvae_onnx.FCVAE;
START APPLICATION fcvae_onnx.FCVAE;
```

### 4.2 Copy data file (after app is running)

```bash
mkdir -p /tmp/fcvae_test
cp data/synthetic_transactions_phase2.csv /tmp/fcvae_test/
```

### 4.3 Monitor

Check that ONNX models loaded successfully:

```bash
grep "FCVAEOnnxScorer" $STRIIM_HOME/logs/striim.server.log | head -20
```

Should show 5 models loaded with their thresholds. Then check output:

```bash
cat /tmp/fcvae_test/scored_output.00
```

---

## 5. Results

With `synthetic_transactions_phase2.csv` (~50,000 rows, ~27.5 hours, 4 combos):

| Metric | Value |
|---|---|
| Combos scored | 4 (Accel_CMP, Accel_nopin, Star_CMP, Star_nopin) |
| Windows per combo | ~3-4 (sliding window emits after 24 hourly counts) |
| Threshold | Per-combo (NLL-based, F1-calibrated) |

### Example JSON output

```json
{
  "combo_key": "Star_CMP",
  "is_anomaly": "false",
  "anomaly_score": "-0.31990835",
  "threshold": "-1.756",
  "window_end": "2025/02/25 23:01:02.000"
}
```

All four combos should appear in the output. With normal synthetic data, scores are above their respective thresholds (not anomalous).

### Result parity with HTTP sidecar

The ONNX scores should match the HTTP sidecar scores within tolerance (max absolute diff < 1e-3 for most windows, < 1.0 for edge cases due to FFT numerical differences). Same anomaly decisions expected.

---

## 6. Teardown and Re-runs

### Stop the application

```sql
USE fcvae_onnx;
STOP APPLICATION fcvae_onnx.FCVAE;
UNDEPLOY APPLICATION fcvae_onnx.FCVAE;
```

### Re-run with same data

```bash
rm -f /tmp/fcvae_test/scored_output*
rm -f /tmp/fcvae_test/synthetic_transactions*.csv
```

Then redeploy and start, and copy data with a new filename:

```sql
DEPLOY APPLICATION fcvae_onnx.FCVAE;
START APPLICATION fcvae_onnx.FCVAE;
```

```bash
cp data/synthetic_transactions_phase2.csv /tmp/fcvae_test/synthetic_transactions_phase2_run2.csv
```

### Full reset

```sql
USE fcvae_onnx;
STOP APPLICATION fcvae_onnx.FCVAE;
UNDEPLOY APPLICATION fcvae_onnx.FCVAE;
DROP APPLICATION fcvae_onnx.FCVAE CASCADE;
DROP TYPE fcvae_onnx.ScorerResult;
DROP TYPE fcvae_onnx.DailyPayloadStream_Type;
USE admin;
UNLOAD OPEN PROCESSOR "/opt/Striim/modules/FCVAEOnnxScorer.scm";
DROP NAMESPACE fcvae_onnx;
```

Stop Striim, then:

```bash
rm -f $STRIIM_HOME/.striim/OpenProcessor/FCVAEOnnxScorer.scm
rm -f $STRIIM_HOME/modules/FCVAEOnnxScorer.scm
rm -f $STRIIM_HOME/lib/fcvae_onnx-types.jar
rm -f /tmp/fcvae_test/scored_output*
rm -f /tmp/fcvae_test/synthetic_transactions*.csv
rm -rf /opt/Striim/fcvae-models
$STRIIM_HOME/bin/server.sh
```

Then start from [Step 2](#2-deploy-to-striim).

---

## Deployment Order (Quick Reference)

```
 1. Copy model directories        cp -r models/fcvae/* /opt/Striim/fcvae-models/
 2. Copy .scm + types JAR         cp to $STRIIM_HOME/modules/ and $STRIIM_HOME/lib/
 3. Stop Striim                   Ctrl+C
 4. Remove types JAR              mv $STRIIM_HOME/lib/fcvae_onnx-types.jar /tmp/
 5. Start Striim                  $STRIIM_HOME/bin/server.sh
 6. CREATE types                  CREATE NAMESPACE fcvae_onnx; USE fcvae_onnx; CREATE TYPE ...
 7. Paste TQL                     CREATE APPLICATION FCVAE; ... END APPLICATION;
 8. Stop Striim                   Ctrl+C
 9. Restore types JAR             mv /tmp/fcvae_onnx-types.jar $STRIIM_HOME/lib/
10. Start Striim                  $STRIIM_HOME/bin/server.sh
11. LOAD OP                       LOAD OPEN PROCESSOR "/opt/Striim/modules/FCVAEOnnxScorer.scm";
12. Wire OP in Flow Designer      DailyPayloadStream -> FCVAEOnnxScorer -> ScorerResultStream
13. Deploy + Start + Data         DEPLOY; START; cp data to /tmp/fcvae_test/
```

---

## Data Flow

```
TxnFileSource (FileReader + DSVParser, watches /tmp/fcvae_test/)
    |
    v
RawTxnStream (WAEvent: data[0]=timestamp, data[1]=network, data[2]=txn_type, ...)
    |
    v
ParseTransactions CQ (builds combo_key = network + "_" + txn_type, parses timestamp)
    |
    v
TypedTxnStream (combo_key, txn_timestamp)
    |
    v
HourlyWindow (1-hour jumping, PARTITION BY combo_key, ON txn_timestamp)
    |
    v
HourlyCounts CQ (COUNT per combo per hour)
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
DailyPayloadStream OF fcvae_onnx.DailyPayloadStream_Type
    |
    v
FCVAEOnnxScorer OP (in-process ONNX Runtime inference)
    |
    v
ScorerResultStream OF fcvae_onnx.ScorerResult
    |
    v
ResultFile (FileWriter + JSONFormatter -> /tmp/fcvae_test/scored_output)
```

---

## Coexistence with HTTP Sidecar

This ONNX deployment (`fcvae_onnx` namespace) is fully independent of the HTTP sidecar deployment (`fcvae` namespace). Both can run on the same Striim instance simultaneously:

| | HTTP Sidecar | ONNX Runtime |
|---|---|---|
| Namespace | `fcvae` | `fcvae_onnx` |
| Application | `fcvae.FCVAE` | `fcvae_onnx.FCVAE` |
| OP module | `FCVAEScoreCaller` | `FCVAEOnnxScorer` |
| Types JAR | `fcvae_types.jar` | `fcvae_onnx-types.jar` |
| Python required | Yes | No |

---

## Environment Reference

| Component | Detail |
|---|---|
| Striim Platform | 5.2.0.4 at `$STRIIM_HOME` |
| Scoring | In-process ONNX Runtime (no external API) |
| Namespace | `fcvae_onnx` |
| Application | `fcvae_onnx.FCVAE` |
| OP module | `FCVAEOnnxScorer` |
| Models dir | `/opt/Striim/fcvae-models/` (configurable via `modelsDir` property) |
| Types JAR | `$STRIIM_HOME/lib/fcvae_onnx-types.jar` (must be present at boot) |
| Data file | `data/synthetic_transactions_phase2.csv` (copy to `/tmp/fcvae_test/` after app starts) |
| Output | `/tmp/fcvae_test/scored_output.00`, `.01`, etc. |
| Striim logs | `$STRIIM_HOME/logs/striim.server.log` |
| Window sizes | 1-hour jumping (aggregation), 24-row sliding (payload assembly) |
| Partitioning | `combo_key` (4 combos: Accel_CMP, Accel_nopin, Star_CMP, Star_nopin) |
| Scoring | NLL-based, F1-calibrated per-combo thresholds, ONNX float32 inference |
| Annotation types | `DailyPayloadStream_Type_1_0` (input), `ScorerResult_1_0` (output) — in `wa.fcvae_onnx` package |
