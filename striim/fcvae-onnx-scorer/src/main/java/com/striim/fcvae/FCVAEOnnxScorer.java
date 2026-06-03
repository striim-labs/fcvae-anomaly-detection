package com.striim.fcvae;

import com.webaction.anno.AdapterType;
import com.webaction.anno.PropertyTemplate;
import com.webaction.anno.PropertyTemplateProperty;
import com.webaction.runtime.components.openprocessor.StriimOpenProcessor;
import com.webaction.runtime.containers.IBatch;
import com.webaction.runtime.containers.WAEvent;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import com.google.gson.Gson;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Objects;

/**
 * FCVAEOnnxScorer -- in-process FCVAE penny-carding anomaly scoring via ONNX Runtime.
 *
 * <p>Replaces the FCVAEScoreCaller HTTP sidecar (POST to a Python FastAPI
 * service) with in-process ONNX inference through ONNX Runtime's Java JNI
 * bindings. No Python dependency, no network hop.
 *
 * <p>This is the PENNY use case: a single pooled model ("Penny_All") that scores
 * a pre-assembled 24-hour window of penny-transaction (amount &lt; $1) hourly
 * counts. The windowing is done UPSTREAM by Striim CQs; the assembled row is
 * converted to a {@code Global.WAEvent} by the WAEUdf {@code createWAEvent} UDF
 * and handed to this pass-through OP. So the OP itself is stateless -- it just
 * scores the 24-vector it receives.
 *
 * <p>This is a WAEvent pass-through Open Processor (the recommended pattern in
 * CLAUDE.md). Input and output are both the runtime WAEvent class, so the OP is
 * wired directly in TQL with {@code CREATE OPEN PROCESSOR ... INSERT INTO ...
 * FROM ...} -- no types JAR, no JAR-removal trick.
 *
 * <p>The incoming WAEvent's data[] (set by the upstream createWAEvent CQ) holds:
 * <pre>
 *   data[0] = combo_key    (String, "Penny_All")
 *   data[1] = values_list  (String, LIST() of 24 hourly penny counts, comma-separated)
 *   data[2] = window_size  (String, "24")
 *   data[3] = window_start (String, timestamp of first hour)
 *   data[4] = window_end   (String, timestamp of last hour)
 * </pre>
 *
 * <p>The OP applies the model's StandardScaler, runs inference to get the
 * per-point NLL, thresholds the last point, and APPENDS the result to data[]
 * (a Striim formatter cannot read userdata, so results must live in data[]):
 * <pre>
 *   data[5] = is_anomaly    (String "true"/"false")
 *   data[6] = anomaly_score (String, last-point NLL)
 *   data[7] = threshold     (String, last_point_threshold)
 * </pre>
 * The same three values are mirrored into userdata for live SysOut. A downstream
 * CQ projects data[0,4,5,6,7] into named fields for the JSONFormatter target.
 */
@PropertyTemplate(
    name = "FCVAEOnnxScorer",
    type = AdapterType.process,
    properties = {
        @PropertyTemplateProperty(name = "ModelDir", type = String.class, required = false,
                defaultValue = "/opt/Striim/fcvae-models/Penny_All"),
        @PropertyTemplateProperty(name = "ValuesIndex", type = Integer.class, required = false,
                defaultValue = "1"),
        @PropertyTemplateProperty(name = "ComboKeyIndex", type = Integer.class, required = false,
                defaultValue = "0"),
        @PropertyTemplateProperty(name = "WindowEndIndex", type = Integer.class, required = false,
                defaultValue = "4"),
        @PropertyTemplateProperty(name = "EnableLogging", type = Boolean.class, required = false,
                defaultValue = "false")
    },
    outputType = com.webaction.proc.events.WAEvent.class,
    inputType  = com.webaction.proc.events.WAEvent.class
)
public class FCVAEOnnxScorer extends StriimOpenProcessor {

    private static final Logger logger = LogManager.getLogger(FCVAEOnnxScorer.class);

    // Configuration (read in start()).
    private String modelDir;
    private int valuesIndex;
    private int comboKeyIndex;
    private int windowEndIndex;
    private boolean enableLogging;

    // ONNX Runtime state, built once in start().
    private OrtEnvironment ortEnv;
    private OrtSession session;
    private ModelConfig config;

    @Override
    public void start() throws Exception {
        super.start();
        final java.util.Map<String, Object> props = getProperties();
        modelDir = orDefault(props.get("ModelDir"), "/opt/Striim/fcvae-models/Penny_All");
        valuesIndex = parseInt(props.get("ValuesIndex"), 1);
        comboKeyIndex = parseInt(props.get("ComboKeyIndex"), 0);
        windowEndIndex = parseInt(props.get("WindowEndIndex"), 4);
        enableLogging = Boolean.parseBoolean(Objects.toString(props.get("EnableLogging"), "false"));
        loadModel();
    }

    /** Loads the single penny ONNX session + config once at startup. */
    private void loadModel() throws IOException, OrtException {
        final File dir = new File(modelDir);
        final File configFile = new File(dir, "model_config.json");
        final File onnxFile = new File(dir, "model.onnx");
        if (!configFile.exists() || !onnxFile.exists()) {
            throw new RuntimeException("Penny model not found in " + modelDir
                + " (need model.onnx + model_config.json)");
        }
        try (FileReader reader = new FileReader(configFile)) {
            config = new Gson().fromJson(reader, ModelConfig.class);
        }
        ortEnv = OrtEnvironment.getEnvironment();
        // ONNX 2.0 keeps weights in a sibling model.onnx.data file; ORT resolves
        // it relative to the .onnx path, so load by absolute path.
        session = ortEnv.createSession(onnxFile.getAbsolutePath());
        logger.info("FCVAEOnnxScorer loaded penny model from {} (threshold={}, mean={}, scale={})",
            modelDir, config.thresholds.last_point_threshold,
            config.scaler.mean, config.scaler.scale);
    }

    @Override
    public void run() {
        final IBatch<WAEvent> batch = getAdded();
        if (batch == null) {
            return;
        }
        for (final WAEvent event : batch) {
            // The SDK WAEvent wraps the runtime WAEvent in its data field.
            final com.webaction.proc.events.WAEvent waevent =
                    (com.webaction.proc.events.WAEvent) event.data;
            if (waevent == null || waevent.data == null) {
                continue;
            }
            try {
                scoreEvent(waevent);
            } catch (final Exception e) {
                logger.error("Error scoring event: {}", e.getMessage(), e);
                continue;   // drop on hard failure rather than emit an unscored window
            }
            send(waevent);
        }
    }

    /** Reads the assembled window from data[], scores it, and appends the result. */
    private void scoreEvent(final com.webaction.proc.events.WAEvent waevent) throws OrtException {
        final Object[] data = waevent.data;
        final String comboKey = stringAt(data, comboKeyIndex, "Penny_All");
        final String windowEnd = stringAt(data, windowEndIndex, "");
        final float[] rawValues = parseValues(stringAt(data, valuesIndex, ""));

        // StandardScaler: (x - mean) / scale.
        final float mean = (float) config.scaler.mean;
        final float scale = (float) config.scaler.scale;
        final float[][][] inputArray = new float[1][1][rawValues.length];
        for (int i = 0; i < rawValues.length; i++) {
            inputArray[0][0][i] = (rawValues[i] - mean) / scale;
        }

        final float lastPointScore;
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnv, inputArray);
             OrtSession.Result result = session.run(
                 Collections.singletonMap(config.onnx.input_name, inputTensor))) {
            final float[][] nll = (float[][]) result.get(config.onnx.output_name).get().getValue();
            lastPointScore = nll[0][nll[0].length - 1];   // last point = scored hour
        }

        final double threshold = config.thresholds.last_point_threshold;
        final boolean isAnomaly = lastPointScore < threshold;

        if (enableLogging) {
            logger.info("Scored combo={} is_anomaly={} score={} threshold={} window_end={}",
                comboKey, isAnomaly, lastPointScore, threshold, windowEnd);
        }
        appendResult(waevent, isAnomaly, lastPointScore, threshold);
    }

    /**
     * Appends the scoring result to data[] (data[5..7]) so a downstream CQ and
     * JSONFormatter can capture it, and mirrors it into userdata for SysOut.
     */
    private void appendResult(final com.webaction.proc.events.WAEvent waevent,
                              final boolean isAnomaly,
                              final float anomalyScore,
                              final double threshold) {
        final String isAnomalyStr = String.valueOf(isAnomaly);
        final String scoreStr = String.valueOf(anomalyScore);
        final String thresholdStr = String.valueOf(threshold);

        final Object[] grown = Arrays.copyOf(waevent.data, waevent.data.length + 3);
        grown[grown.length - 3] = isAnomalyStr;
        grown[grown.length - 2] = scoreStr;
        grown[grown.length - 1] = thresholdStr;
        waevent.data = grown;

        if (waevent.userdata == null) {
            waevent.userdata = new HashMap<>();
        }
        waevent.userdata.put("is_anomaly", isAnomalyStr);
        waevent.userdata.put("anomaly_score", scoreStr);
        waevent.userdata.put("threshold", thresholdStr);
    }

    private static String stringAt(final Object[] data, final int idx, final String fallback) {
        if (idx >= 0 && idx < data.length && data[idx] != null) {
            return String.valueOf(data[idx]).trim();
        }
        return fallback;
    }

    /**
     * Parses the LIST() values into a float[]. Striim's LIST() yields a
     * comma-separated string, optionally bracketed and space-padded
     * ("[100, 102, ...]" or "100, 102, ..."); both are handled.
     */
    private static float[] parseValues(final String raw) {
        final String cleaned = raw.replaceAll("[\\[\\]]", "").trim();
        if (cleaned.isEmpty()) {
            return new float[0];
        }
        final String[] parts = cleaned.split(",");
        final float[] values = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Float.parseFloat(parts[i].trim());
        }
        return values;
    }

    private static String orDefault(final Object v, final String dflt) {
        final String s = Objects.toString(v, dflt);
        return s.isEmpty() ? dflt : s;
    }

    private static int parseInt(final Object v, final int dflt) {
        try {
            return Integer.parseInt(Objects.toString(v, String.valueOf(dflt)).trim());
        } catch (final NumberFormatException e) {
            return dflt;
        }
    }

    @Override
    public void close() throws Exception {
        super.close();
        if (session != null) {
            try {
                session.close();
            } catch (final OrtException e) {
                logger.error("Error closing ONNX session: {}", e.getMessage());
            }
        }
        if (ortEnv != null) {
            ortEnv.close();
        }
        logger.info("FCVAEOnnxScorer closed.");
    }

    // Stateless: the window is assembled upstream in Striim CQs.
    @Override
    public java.util.Map getAggVec() { return null; }

    @Override
    public void setAggVec(final java.util.Map aggVec) { }
}
