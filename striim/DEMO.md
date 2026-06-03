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

## 1. Clean up before a run

FileReader tracks each file by name + byte offset, so a file it has already read
will not be re-read. Clear the watch dir (both prior inputs and the scored
output) so the next run starts fresh:

```bash
rm -f /tmp/fcvae_onnx_penny_test/synthetic_transactions*.csv
rm -f /tmp/fcvae_onnx_penny_test/penny_scored_output*
# (the dir itself stays; the app keeps watching it)
mkdir -p /tmp/fcvae_onnx_penny_test
```

The app can stay deployed and running across runs -- you only need to manage the
files in the watch dir.

## 2. Start the demo: copy a fresh file into the watch path

Copy the source CSV in with a **unique name each run** (so FileReader treats it
as new). The wildcard `synthetic_transactions*.csv` matches any suffix:

```bash
cp data/synthetic_transactions.csv \
   /tmp/fcvae_onnx_penny_test/synthetic_transactions_$(date +%s).csv
```

> Copy the file in *after* the app is deployed and running. A file already
> present when the app starts may be treated as already processed.

If the app is not running yet:

```sql
DEPLOY APPLICATION fcvae_onnx.FCVAEOnnxPenny;
START  APPLICATION fcvae_onnx.FCVAEOnnxPenny;
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

## 4. Reset between demos

To re-run cleanly, repeat steps 1-3. To rebuild the OP after a code change:

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

Stop the Python FastAPI scoring service entirely, then run steps 1-3. The
pipeline keeps scoring -- inference runs in-process in the Striim JVM via ONNX
Runtime, with no external service.
