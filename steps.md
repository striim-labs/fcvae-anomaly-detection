# FCVAE In-Process ONNX Pipeline -- Local Run Instructions

Prereqs: Striim Platform **5.2.0.4** installed at `$STRIIM_HOME` (e.g. `/opt/Striim`,
OpenJDK 11), and Maven + JDK 11 **only if you want to rebuild the OP** (the repo already
ships a built `.scm`).

## 1. Clone and check out the branch

```bash
git clone https://github.com/striim-labs/fcvae-anomaly-detection.git
cd fcvae-anomaly-detection
git checkout onnx-hybrid-passthrough
```

## 2. Install the WAEUdf library (one-time, required)

The `createWAEvent` UDF is **not installed by default**. Without it the `PennyToWaevent`
CQ will not compile.

1. Download from https://webaction.atlassian.net/wiki/spaces/SE/pages/2318336001
2. Copy the JAR into `$STRIIM_HOME/lib/` and **restart Striim**.
3. Confirm:

```bash
find $STRIIM_HOME/lib -name '*.jar' -exec sh -c \
  'jar tf "{}" 2>/dev/null | grep -q "com/webaction/helpers/WAEUdf" && echo "FOUND in {}"' \;
```

## 3. Stage the Open Processor module

The repo carries a prebuilt artifact at `striim/fcvae-onnx-scorer/FCVAEOnnxScorer.scm`.
Copy it into Striim's upload area:

```bash
cp striim/fcvae-onnx-scorer/FCVAEOnnxScorer.scm "$STRIIM_HOME/UploadedFiles/"
```

## 4. Stage the model bundle and create the I/O dir

```bash
mkdir -p "$STRIIM_HOME/fcvae-models"
cp -r models/fcvae/Penny_All "$STRIIM_HOME/fcvae-models/"   # model.onnx + model.onnx.data + model_config.json
mkdir -p /tmp/fcvae_onnx_penny_test                          # FileReader watch / output dir
```

The TQL hardcodes `/opt/Striim/fcvae-models/Penny_All`. If your `$STRIIM_HOME` is not
`/opt/Striim`, either symlink it or edit `ModelDir` in `striim/FCVAE_ONNX.tql`.

## 5. Load, deploy, start (in the Striim console)

```sql
LOAD OPEN PROCESSOR 'UploadedFiles/FCVAEOnnxScorer.scm';
LIST OPENPROCESSORS;          -- confirm FCVAEOnnxScorer is listed
```

Then import `striim/FCVAE_ONNX.tql` (creates namespace `fcvae_onnx`, the `PennyPayloadType`,
and app `FCVAEOnnxPenny`), and:

```sql
DEPLOY APPLICATION fcvae_onnx.FCVAEOnnxPenny;
START  APPLICATION fcvae_onnx.FCVAEOnnxPenny;
```

## 6. Feed the data (after START, so FileReader sees a fresh arrival)

```bash
cp data/synthetic_transactions.csv /tmp/fcvae_onnx_penny_test/
```

## 7. Check the output

```bash
tail -f /tmp/fcvae_onnx_penny_test/penny_scored_output*
grep -c '"window_end"'          /tmp/fcvae_onnx_penny_test/penny_scored_output*   # windows scored
grep -c '"is_anomaly":"true"'   /tmp/fcvae_onnx_penny_test/penny_scored_output*   # anomalies tripped
```

Each record:

```json
{"combo_key":"Penny_All","window_end":"2025/02/17 00:58:09.000","is_anomaly":"false","anomaly_score":"-0.6986134","threshold":"-21.3322"}
```

`is_anomaly` is `true` when `anomaly_score < threshold`. The first 24 completed hours
produce no output (the model needs a full 24-hour window before it scores).


