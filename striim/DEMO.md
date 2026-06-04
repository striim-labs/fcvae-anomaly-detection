# FCVAE Penny ONNX Demo -- Run Loop

Quick operational runbook for the in-process ONNX penny pipeline
(`fcvae_onnx.FCVAEOnnxPenny`). For first-time setup (WAEUdf install, model
staging, LOAD/DEPLOY), see [../STRIIM_ONNX.md](../STRIIM_ONNX.md). This file is
just the repeatable demo loop.

## Paths at a glance

| Thing | Path |
|---|---|
| FileReader watch dir | `/tmp/fcvae_onnx_penny_test/` |
| Input wildcard | `synthetic_transactions*.csv` |
| Source data file | `data/synthetic_transactions.csv` (repo) |
| Scored output | `/tmp/fcvae_onnx_penny_test/penny_scored_output*` |
| Model bundle | `/opt/Striim/fcvae-models/Penny_All/` |

## 1. Run / re-run (fixed filename)

FileReader tracks each file by name + byte offset, so re-copying the **same**
filename into a still-running app is NOT re-read. To re-run with the same
filename, bounce the app so the read position resets, clear the watch dir, then
copy the file in *after* START (an empty dir at start means the file is seen as
a fresh arrival):

```sql
-- in the Striim console: reset the app (skip if it isn't deployed yet)
STOP     APPLICATION fcvae_onnx.FCVAEOnnxPenny;
UNDEPLOY APPLICATION fcvae_onnx.FCVAEOnnxPenny;
```

```bash
# clear prior input + scored output
rm -f /tmp/fcvae_onnx_penny_test/synthetic_transactions.csv
rm -f /tmp/fcvae_onnx_penny_test/penny_scored_output*
mkdir -p /tmp/fcvae_onnx_penny_test
```

```sql
DEPLOY APPLICATION fcvae_onnx.FCVAEOnnxPenny;
START  APPLICATION fcvae_onnx.FCVAEOnnxPenny;
```

```bash
# copy the SAME file in AFTER start
cp data/synthetic_transactions.csv /tmp/fcvae_onnx_penny_test/
```

## 3. Check the scored output

```bash
# Tail the latest scored windows as they are written:
tail -f /tmp/fcvae_onnx_penny_test/penny_scored_output*

# Count how many windows were scored:
grep -c '"window_end"' /tmp/fcvae_onnx_penny_test/penny_scored_output*

# Did any window trip the anomaly threshold?
grep -c '"is_anomaly":"true"' /tmp/fcvae_onnx_penny_test/penny_scored_output*

# Show the anomalous windows (if any):
grep -B2 -A3 '"is_anomaly":"true"' /tmp/fcvae_onnx_penny_test/penny_scored_output*
```

Each record looks like:

```json
{
  "combo_key":"Penny_All",
  "window_end":"2025/02/17 00:58:09.000",
  "is_anomaly":"false",
  "anomaly_score":"-0.6986134",
  "threshold":"-21.3322"
}
```

`anomaly_score` is the last-point NLL; `is_anomaly` is `true` when
`anomaly_score < threshold`. The first 24 completed hours produce no output
(the model needs a full 24-hour window before it scores).

## 4. Rebuild the OP (after a code change)

To re-run the demo, just repeat step 1. To rebuild the OP itself after a code
change:

```sql
STOP     APPLICATION fcvae_onnx.FCVAEOnnxPenny;
UNDEPLOY APPLICATION fcvae_onnx.FCVAEOnnxPenny;
UNLOAD OPEN PROCESSOR 'UploadedFiles/FCVAEOnnxScorer.scm';
```
```bash
striim/fcvae-onnx-scorer/build.sh   # rebuilds + restages the .scm
```
Then do a full Striim restart before `LOAD`-ing again (ZLIB OpenProcessor-cache,
per CLAUDE.md), re-import the TQL, and DEPLOY/START.

## No-API proof (the demo's punchline)

Stop the Python FastAPI scoring service entirely, then run step 1. The pipeline
keeps scoring -- inference runs in-process in the Striim JVM via ONNX Runtime,
with no external service.

---

# Demo Script (narration)

A walkthrough you can read aloud while the pipeline runs.

## The model: FCVAE

The detector is a Frequency-enhanced Conditional Variational Autoencoder (FCVAE,
Wang et al., WWW 2024). Like any variational autoencoder it learns to compress a
signal into a small latent code and reconstruct it, and it flags anything it
reconstructs poorly as anomalous. What makes FCVAE special is that it conditions
on the signal's frequency content: an FFT-based Global Frequency Module looks at
the whole 24-hour window and a Local Frequency Module looks at short sub-windows,
so the model learns the *periodic shape* of normal traffic (the daily rhythm of
when transactions happen) rather than just an average level. At scoring time it
reconstructs the window and reports a per-point negative-log-likelihood (NLL);
the final hour's NLL is the anomaly signal. We score deterministically by using
the posterior mean ($z = \mu$) instead of sampling, which is what makes the model
exportable to ONNX.

## The problem: penny carding

Card "carding" attacks validate stolen card numbers by running a burst of tiny
authorizations, typically under \$1, before the numbers are used for real fraud.
Individually those penny transactions look harmless; the tell is an abnormal
*spike in their hourly frequency*. So we pool every penny transaction (amount
$< \$1$) across all networks and card types into one signal, "Penny_All", count
it per hour, and ask the model whether the last hour's count breaks the learned
daily pattern.

## The pipeline, component by component

* **PennyTxnFileSource** (FileReader + DSVParser): reads raw transaction rows
  from CSV and emits them as native `Global.WAEvent`s, one event per transaction,
  with the columns in `data[]`.
* **PennyParseTxn** (CQ): pulls the two fields we need out of `data[]` -- the
  event timestamp (`data[0]`) and the dollar amount (`data[3]`) -- into typed
  fields.
* **PennyFilter** (CQ): keeps only penny transactions (`amount < 1.00`) and tags
  them all with the constant key `Penny_All`, pooling every network/type into one
  stream.
* **PennyHourlyWindow + PennyHourlyCounts** (jumping window + CQ): groups the
  penny transactions into 1-hour event-time buckets and emits one count per hour.
* **PennyDailyWindow + PennyAssembleDailyPayload** (sliding window + CQ): holds
  the last 24 hourly counts and assembles them into a single row whose
  `values_list` is the comma-separated 24-number vector (it only emits once 24
  hours are present).
* **PennyToWaevent** (CQ): wraps that aggregated row back into a `Global.WAEvent`
  using the `WAEUdf.createWAEvent` UDF, so the five payload fields land in
  `data[0..4]` and the scorer can consume them as a pass-through event.
* **PennyOnnxProc** (the Open Processor): scores the 24-vector with the ONNX model
  in-process and appends the result to `data[]` (see below).
* **PennyFormatScored** (CQ): projects the result fields out of `data[]` into
  named columns (a formatter cannot read the event's userdata).
* **PennyResultFile** (FileWriter + JSONFormatter): writes one JSON record per
  scored window.

## How the data changes shape

Each stage narrows and reshapes the stream:
`raw transactions` -> (filter) `penny transactions only` -> (hourly window)
`one count per hour` -> (24-row window) `a 24-number vector per hour` ->
(OP) `that vector + an anomaly verdict`. We go from millions of individual rows
to one scored decision per hour.

## What the Open Processor does, and how ONNX is embedded

The OP, `FCVAEOnnxScorer`, loads the model once at startup: it reads
`model.onnx` plus `model_config.json` from `/opt/Striim/fcvae-models/Penny_All/`
and opens an ONNX Runtime session. ONNX Runtime ships as a Java/JNI library that
is bundled directly inside the OP's `.scm` fat JAR, so the C++ inference engine
runs *inside the Striim JVM* -- there is no Python, no HTTP, no separate service.
Per event the OP reads the 24-count vector from `data[1]`, normalizes it with the
training `StandardScaler` ($(x - \mu)/\sigma$ using the mean and scale baked into
`model_config.json`), shapes it into a $(1, 1, 24)$ tensor, runs
`OrtSession.run()`, and reads back the per-point NLL. It takes the last point as
the anomaly score, compares it to the model's calibrated threshold, and appends
`is_anomaly`, `anomaly_score`, and `threshold` onto the event. This is the exact
work the old FastAPI sidecar did over HTTP, now done in a single in-memory call.

## Reading the output

```json
{
  "combo_key":"Penny_All",
  "window_end":"2025/02/17 00:58:09.000",
  "is_anomaly":"false",
  "anomaly_score":"-0.6986134",
  "threshold":"-21.3322"
}
```

* **`combo_key`** -- which model scored this window. For the penny use case it is
  always `Penny_All` (the single pooled model).
* **`window_end`** -- the timestamp of the final, scored hour of the 24-hour
  window. Each successive record advances by one hour as the window slides.
* **`anomaly_score`** -- the model's reconstruction log-likelihood at that final
  hour. Closer to $0$ means the hour fit the learned daily pattern well; a large
  negative value means the model could not explain it (a frequency anomaly). Here
  $-0.70$ is a clean, normal hour.
* **`threshold`** -- the calibrated cutoff for this model (the low percentile of
  normal validation scores). An hour is flagged only when its score falls *below*
  this line.
* **`is_anomaly`** -- the verdict: `true` when `anomaly_score < threshold`. With
  a score of $-0.70$ against a threshold of $-21.33$, this window is far inside
  normal, so `false`. A carding spike would drive the score sharply negative,
  past $-21.33$, and flip this to `true`.
