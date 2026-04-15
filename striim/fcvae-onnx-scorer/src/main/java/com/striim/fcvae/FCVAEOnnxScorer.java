package com.striim.fcvae;

import com.webaction.anno.AdapterType;
import com.webaction.anno.PropertyTemplate;
import com.webaction.anno.PropertyTemplateProperty;
import com.webaction.runtime.components.openprocessor.StriimOpenProcessor;
import com.webaction.runtime.containers.WAEvent;
import com.webaction.runtime.containers.IBatch;
import wa.fcvae_onnx.ScorerResult_1_0;
import wa.fcvae_onnx.DailyPayloadStream_Type_1_0;

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
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * Striim Open Processor that scores FCVAE windows using ONNX Runtime.
 *
 * Replaces FCVAEScoreCaller (HTTP sidecar) with in-process ONNX inference.
 * Loads .onnx model files and model_config.json directly, runs inference
 * via ONNX Runtime's Java JNI bindings. No Python dependency.
 *
 * Input WAEvent.data is expected to be an Object with fields accessible
 * as an array. The upstream CQ should produce events with:
 *   data[0] = combo_key    (String, e.g. "Accel_CMP")
 *   data[1] = values_list  (String, LIST() of 24 hourly counts, comma-separated)
 *   data[2] = window_size  (Integer, always 24)
 *   data[3] = window_start (String, timestamp of first hour)
 *   data[4] = window_end   (String, timestamp of last hour)
 *
 * Output is a typed ScorerResult_1_0 with:
 *   combo_key, is_anomaly, anomaly_score, threshold, window_end
 */
@PropertyTemplate(
    name = "FCVAEOnnxScorer",
    type = AdapterType.process,
    properties = {
        @PropertyTemplateProperty(
            name = "modelsDir",
            type = String.class,
            required = false,
            defaultValue = "/opt/Striim/fcvae-models"
        )
    },
    outputType = DailyPayloadStream_Type_1_0.class,
    inputType = ScorerResult_1_0.class
)
public class FCVAEOnnxScorer extends StriimOpenProcessor {

    private static final Logger logger = LogManager.getLogger(FCVAEOnnxScorer.class);
    private static final String[] MODEL_NAMES = {
        "Penny_All", "Accel_CMP", "Accel_nopin", "Star_CMP", "Star_nopin"
    };
    private static final String DEFAULT_MODEL = "Penny_All";

    private String modelsDir;
    private OrtEnvironment ortEnv;
    private Map<String, LoadedModel> models;

    private static class LoadedModel {
        final OrtSession session;
        final ModelConfig config;

        LoadedModel(OrtSession session, ModelConfig config) {
            this.session = session;
            this.config = config;
        }
    }

    @Override
    public void run() {
        if (models == null) {
            initialize();
        }

        IBatch<WAEvent> batch = getAdded();
        if (batch == null) return;

        Iterator<WAEvent> it = batch.iterator();
        while (it.hasNext()) {
            WAEvent inputEvent = it.next();

            try {
                Object eventData = inputEvent.data;
                String comboKey;
                String valuesJson;
                String windowEnd;

                // Unwrap nested WAEvent if needed
                if (eventData != null && eventData.getClass().getName().contains("WAEvent")) {
                    try {
                        java.lang.reflect.Field dataField = eventData.getClass().getField("data");
                        Object innerData = dataField.get(eventData);
                        logger.info("Unwrapped nested WAEvent ({}), inner data type: {}",
                            eventData.getClass().getName(),
                            innerData != null ? innerData.getClass().getName() : "null");
                        eventData = innerData;
                    } catch (NoSuchFieldException | IllegalAccessException e) {
                        logger.error("Failed to unwrap WAEvent: {}", e.getMessage());
                    }
                }

                if (eventData instanceof Object[]) {
                    Object[] arr = (Object[]) eventData;
                    comboKey = arr.length > 0 ? String.valueOf(arr[0]).trim() : "";
                    String rawValues = arr.length > 1 ? String.valueOf(arr[1]).trim() : "[]";
                    windowEnd = arr.length > 4 ? String.valueOf(arr[4]).trim() : "";
                    valuesJson = normalizeValuesList(rawValues);
                } else {
                    try {
                        java.lang.reflect.Field f0 = eventData.getClass().getField("combo_key");
                        java.lang.reflect.Field f1 = eventData.getClass().getField("values_list");
                        java.lang.reflect.Field f2 = eventData.getClass().getField("window_end");
                        comboKey = String.valueOf(f0.get(eventData)).trim();
                        String rawValues = String.valueOf(f1.get(eventData)).trim();
                        windowEnd = String.valueOf(f2.get(eventData)).trim();
                        valuesJson = normalizeValuesList(rawValues);
                    } catch (NoSuchFieldException e) {
                        logger.error("Cannot extract fields from event data of type: {}",
                            eventData.getClass().getName());
                        continue;
                    }
                }

                // Map combo_key to model name; unrecognized combos fall back to Penny_All
                String modelName = models.containsKey(comboKey) ? comboKey : DEFAULT_MODEL;
                LoadedModel loaded = models.get(modelName);
                if (loaded == null) {
                    logger.error("No model loaded for combo={}, modelName={}", comboKey, modelName);
                    continue;
                }

                // Parse values and normalize
                float[] rawValues = parseValues(valuesJson);
                float[] normalized = new float[rawValues.length];
                float mean = (float) loaded.config.scaler.mean;
                float scale = (float) loaded.config.scaler.scale;
                for (int i = 0; i < rawValues.length; i++) {
                    normalized[i] = (rawValues[i] - mean) / scale;
                }

                // Create tensor of shape (1, 1, 24)
                float[][][] inputArray = new float[1][1][normalized.length];
                System.arraycopy(normalized, 0, inputArray[0][0], 0, normalized.length);

                // Run ONNX inference
                OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnv, inputArray);
                try {
                    OrtSession.Result result = loaded.session.run(
                        Collections.singletonMap(loaded.config.onnx.input_name, inputTensor));
                    try {
                        float[][] nllScores = (float[][]) result.get(loaded.config.onnx.output_name)
                            .get().getValue();
                        float lastPointScore = nllScores[0][23];
                        boolean isAnomaly = lastPointScore < loaded.config.thresholds.last_point_threshold;

                        logger.info("Scored combo={}, model={}, is_anomaly={}, score={}, threshold={}",
                            comboKey, modelName, isAnomaly, lastPointScore,
                            loaded.config.thresholds.last_point_threshold);

                        ScorerResult_1_0 scorerResult = new ScorerResult_1_0();
                        scorerResult.combo_key = comboKey;
                        scorerResult.is_anomaly = String.valueOf(isAnomaly);
                        scorerResult.anomaly_score = String.valueOf(lastPointScore);
                        scorerResult.threshold = String.valueOf(loaded.config.thresholds.last_point_threshold);
                        scorerResult.window_end = windowEnd;

                        send(scorerResult);
                    } finally {
                        result.close();
                    }
                } finally {
                    inputTensor.close();
                }

            } catch (Exception e) {
                logger.error("Error processing event: {}", e.getMessage(), e);
            }
        }
    }

    private void initialize() {
        if (modelsDir == null || modelsDir.isEmpty()) {
            modelsDir = "/opt/Striim/fcvae-models";
            logger.warn("modelsDir was null/empty, using default: {}", modelsDir);
        }

        try {
            ortEnv = OrtEnvironment.getEnvironment();
            logger.info("OrtEnvironment created successfully.");
        } catch (Exception e) {
            throw new RuntimeException("Failed to create OrtEnvironment", e);
        }

        models = new HashMap<>();
        Gson gson = new Gson();

        for (String modelName : MODEL_NAMES) {
            File modelDir = new File(modelsDir, modelName);
            File configFile = new File(modelDir, "model_config.json");
            File onnxFile = new File(modelDir, "model.onnx");

            if (!configFile.exists() || !onnxFile.exists()) {
                logger.warn("Skipping model {}: config={} onnx={}",
                    modelName, configFile.exists(), onnxFile.exists());
                continue;
            }

            try {
                ModelConfig config;
                try (FileReader reader = new FileReader(configFile)) {
                    config = gson.fromJson(reader, ModelConfig.class);
                }

                OrtSession session = ortEnv.createSession(onnxFile.getAbsolutePath());
                models.put(modelName, new LoadedModel(session, config));

                logger.info("Loaded model: {} (threshold={}, mean={}, scale={})",
                    modelName,
                    config.thresholds.last_point_threshold,
                    config.scaler.mean,
                    config.scaler.scale);

            } catch (OrtException | IOException e) {
                logger.error("Failed to load model {}: {}", modelName, e.getMessage(), e);
            }
        }

        logger.info("FCVAEOnnxScorer initialized. modelsDir={}, models loaded: {}/{}",
            modelsDir, models.size(), MODEL_NAMES.length);

        if (models.isEmpty()) {
            throw new RuntimeException("No ONNX models loaded from " + modelsDir);
        }
    }

    private float[] parseValues(String valuesJson) {
        String cleaned = valuesJson.replaceAll("[\\[\\]]", "").trim();
        String[] parts = cleaned.split(",");
        float[] values = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Float.parseFloat(parts[i].trim());
        }
        return values;
    }

    private String normalizeValuesList(String raw) {
        if (raw == null || raw.isEmpty()) return "[]";
        String cleaned = raw.trim();
        if (cleaned.startsWith("[")) cleaned = cleaned.substring(1);
        if (cleaned.endsWith("]"))   cleaned = cleaned.substring(0, cleaned.length() - 1);
        String[] parts = cleaned.split(",");
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < parts.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(parts[i].trim());
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public void close() throws Exception {
        logger.info("FCVAEOnnxScorer shutting down.");
        if (models != null) {
            for (Map.Entry<String, LoadedModel> entry : models.entrySet()) {
                try {
                    entry.getValue().session.close();
                    logger.info("Closed ONNX session for model: {}", entry.getKey());
                } catch (OrtException e) {
                    logger.error("Error closing session for {}: {}", entry.getKey(), e.getMessage());
                }
            }
        }
        if (ortEnv != null) {
            ortEnv.close();
        }
    }

    @Override
    public Map getAggVec() { return null; }

    @Override
    public void setAggVec(Map aggVec) { }
}
