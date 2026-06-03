# Striim FCVAE Penny-Carding Pipeline: In-Process ONNX Setup Guide

**Striim Version:** Platform 5.2.0.4 (OpenJDK 11)
**Use case:** Penny carding -- a single pooled `Penny_All` FCVAE model scoring the hourly frequency of penny transactions (amount < $1)
**Namespace:** `fcvae_onnx` (fully isolated from the HTTP sidecar's `fcvae` namespace)
**Pipeline:** FileReader -> Parse CQ -> Penny filter (amount < $1) -> Hourly Jumping Window -> 24-Hour Sliding Window -> `WAEUdf.createWAEvent` CQ -> Open Processor (ONNX Runtime) -> Format CQ -> FileWriter (JSON)

This deploys the penny-carding pipeline scoring **in-process via ONNX Runtime's Java JNI bindings**: the `Penny_All` model runs inside the Striim JVM, with no Python dependency and no FastAPI sidecar. It is the in-process counterpart to the HTTP sidecar penny pipeline (`STRIIM_FCVAE.md`, namespace `fcvae`); both can coexist on the same Striim instance, and deploying this one never touches the running `fcvae` app.

The windowing stays in **Striim CQs** (penny filter -> hourly count -> 24-row sliding window). The assembled 24-hour row is converted to a `Global.WAEvent` by the **WAEUdf `createWAEvent` UDF**, so the scorer is a pure **WAEvent pass-through** Open Processor wired **directly in TQL**. That means:

- **No types JAR** and **no JAR-removal trick** for the OP (it is a pass-through OP).
- **No Flow Designer** -- the OP is wired in TQL with `CREATE OPEN PROCESSOR ... INSERT INTO ... FROM ...`.
- **One `CREATE TYPE`** is required (the payload type for `createWAEvent`), and the **WAEUdf library** must be installed.

> Why the createWAEvent UDF: a Striim CQ cannot turn aggregate columns into a `Global.WAEvent` directly (that type has 7 fixed fields -- `data`, `metadata`, `userdata`, ...). The supported way for a CQ to emit a WAEvent is the WAEUdf library's `createWAEvent(typeName, tableName, operation, data[], before[])`, which packages a value array into a WAEvent with a declared Striim type attached. See the Confluence pages "Reconstruct WAEvents (from IBR, File, Queue or test)" and "JsonNodeEvent to WAEvent conversion."

> Single model. This is the penny use case (one pooled `Penny_All` model). The multi-combo volume use case (per network/transaction-type models) is covered in the notebooks and is not what this pipeline deploys.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install the WAEUdf Library](#2-install-the-waeudf-library)
3. [Build and Stage the Open Processor](#3-build-and-stage-the-open-processor)
4. [Stage the Model and Data](#4-stage-the-model-and-data)
5. [Load, Deploy, and Run](#5-load-deploy-and-run)
6. [Verify](#6-verify)
7. [Teardown and Re-runs](#7-teardown-and-re-runs)
8. [Retraining / Model Updates](#8-retraining--model-updates)

---

## 1. Prerequisites

| Requirement | Detail |
|---|---|
| Striim Platform | 5.2.0.4, installed at `$STRIIM_HOME` (e.g. `/opt/Striim`), OpenJDK 11 |
| WAEUdf library | The WAEvent UDF add-on JAR (provides `com.webaction.helpers.WAEUdf`) -- see step 2 |
| Maven + JDK 11 | to build the Open Processor |
| Penny model bundle | `models/fcvae/Penny_All/` (`model.onnx` + `model.onnx.data` + `model_config.json`), from `code/5_export_onnx.py` |
| Penny data | `data/synthetic_transactions.csv` (7-column penny dataset incl. `amount`) |

The runtime WAEvent class (`com.webaction.proc.events.WAEvent`) lives in `$STRIIM_HOME/lib/Common-5.2.0.4.jar`; the build resolves it system-scoped from the install (see `pom.xml`). ONNX Runtime's native libraries are bundled inside the `.scm` fat JAR, so nothing else is needed on the host for the OP itself.

---

## 2. Install the WAEUdf Library

The `createWAEvent` UDF that converts the aggregated row to a `Global.WAEvent` comes from the WAEUdf library, which is **not installed by default**.

1. Download the WAEUdf binaries: https://webaction.atlassian.net/wiki/spaces/SE/pages/2318336001
2. Copy the JAR into `$STRIIM_HOME/lib/`.
3. Restart Striim so the class loads.
4. Confirm it is on the classpath:

```bash
find $STRIIM_HOME/lib -name '*.jar' -exec sh -c \
  'jar tf "{}" 2>/dev/null | grep -q "com/webaction/helpers/WAEUdf" && echo "FOUND in {}"' \;
```

If this prints nothing, the UDF is not installed and the `PennyToWaevent` CQ will fail to compile.

---

## 3. Build and Stage the Open Processor

```bash
cd striim/fcvae-onnx-scorer
./build.sh
```

This runs `mvn clean package` (shading ONNX Runtime + Gson into the fat JAR), clears any stale cache copy, and stages the module at `$STRIIM_HOME/UploadedFiles/FCVAEOnnxScorer.scm` (plus a committed repo copy).

Sanity-check the artifact:

```bash
unzip -p $STRIIM_HOME/UploadedFiles/FCVAEOnnxScorer.scm META-INF/MANIFEST.MF | grep Striim
unzip -l $STRIIM_HOME/UploadedFiles/FCVAEOnnxScorer.scm | grep -E 'ai/onnxruntime|wa/fcvae_onnx' | head
```

Expect the three `Striim-*` manifest entries and `ai/onnxruntime/...` classes, and **no** `wa/fcvae_onnx/...` entries.

---

## 4. Stage the Model and Data

```bash
# Single penny model bundle:
mkdir -p /opt/Striim/fcvae-models
cp -r models/fcvae/Penny_All /opt/Striim/fcvae-models/

# Isolated input/output dir (NOT the production app's dir):
mkdir -p /tmp/fcvae_onnx_penny_test
```

`/opt/Striim/fcvae-models/Penny_All/` must contain `model.onnx`, `model.onnx.data`, and `model_config.json`. Do not copy the CSV in yet (FileReader processes files already present at start; copy it after the app is running -- see step 5).

---

## 5. Load, Deploy, and Run

In the Striim console:

```sql
-- 1. Load the pass-through OP (registers in the Global namespace).
LOAD OPEN PROCESSOR 'UploadedFiles/FCVAEOnnxScorer.scm';
LIST OPENPROCESSORS;   -- confirm FCVAEOnnxScorer is listed

-- 2. Import striim/FCVAE_ONNX.tql. It creates namespace fcvae_onnx, the
--    PennyPayloadType, and app FCVAEOnnxPenny. Then:
DEPLOY APPLICATION fcvae_onnx.FCVAEOnnxPenny;
START  APPLICATION fcvae_onnx.FCVAEOnnxPenny;
```

Then feed data (after the app is running, so FileReader picks it up):

```bash
cp data/synthetic_transactions.csv /tmp/fcvae_onnx_penny_test/
```

---

## 6. Verify

```bash
# Scored windows land here as JSON:
ls -la /tmp/fcvae_onnx_penny_test/penny_scored_output*
cat    /tmp/fcvae_onnx_penny_test/penny_scored_output*
```

Each record carries `combo_key` (always `Penny_All`), `window_end`, `is_anomaly`, `anomaly_score`, `threshold`. In the Striim console, confirm:

- The OP loaded the penny model (`FCVAEOnnxScorer loaded penny model from ...`).
- The `PennyToWaevent` CQ compiled (i.e. WAEUdf is installed) and `PennyDailyPayloadStream` carries events.
- The production app is untouched: `LIST APPLICATIONS;` shows `fcvae.FCVAE` still in its prior state.
- **No-API proof:** stop the FastAPI service entirely -- this pipeline keeps scoring, because inference is in-process.

Parity sanity: for a few windows, the `is_anomaly` / `anomaly_score` should match the `fcvae` HTTP penny app on the same input, consistent with the 99.9%+ PyTorch-vs-ONNX decision agreement validated by `code/5_export_onnx.py`.

---

## 7. Teardown and Re-runs

```sql
STOP     APPLICATION fcvae_onnx.FCVAEOnnxPenny;
UNDEPLOY APPLICATION fcvae_onnx.FCVAEOnnxPenny;
-- To rebuild the OP: UNLOAD, restage, reLOAD (full Striim restart before reload
-- avoids the ZLIB OpenProcessor-cache issue noted in CLAUDE.md).
UNLOAD OPEN PROCESSOR 'UploadedFiles/FCVAEOnnxScorer.scm';
```

FileReader tracks files by name + offset. To re-feed, use a fresh filename suffix (e.g. `synthetic_transactions_run2.csv`; the wildcard `synthetic_transactions*.csv` already matches it) or clear `/tmp/fcvae_onnx_penny_test/` between runs.

---

## 8. Retraining / Model Updates

Training is unchanged (Python). Re-export with `code/5_export_onnx.py` to produce a new penny bundle, stage it under `/opt/Striim/fcvae-models/Penny_All/` (optionally version it as `Penny_All/vN/` and point `ModelDir` at the new path), and restart the app to pick it up. The scorer loads the model once in `start()`.

> Roadmap note: Striim has a funded functional spec ("ML Inferencing Pipelines," ticket DEV-58023) for first-class in-process ONNX via `CREATE MODEL ... USING FileProvider(...)` + an `INFER USING model(...)` CQ clause -- which would replace this OP + createWAEvent plumbing with native Tungsten syntax. This pipeline is a working precursor of that design.
