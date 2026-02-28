package com.striim.fcvae;

import com.webaction.anno.AdapterType;
import com.webaction.anno.PropertyTemplate;
import com.webaction.anno.PropertyTemplateProperty;
import com.webaction.runtime.components.openprocessor.StriimOpenProcessor;
import com.webaction.runtime.containers.WAEvent;
import com.webaction.runtime.containers.IBatch;
import wa.fcvae.ScorerResult_1_0;
import wa.fcvae.DailyPayloadStream_Type_1_0;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Iterator;
import java.util.Map;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Striim Open Processor that calls an external FCVAE scoring API.
 *
 * Input WAEvent.data is expected to be an Object with fields accessible
 * as an array. The upstream CQ should produce events with:
 *   data[0] = combo_key    (String, e.g. "Accel_CMP")
 *   data[1] = values_list  (String, LIST() of 24 hourly counts, comma-separated)
 *   data[2] = window_size  (Integer, always 24)
 *   data[3] = window_start (String, timestamp of first hour)
 *   data[4] = window_end   (String, timestamp of last hour)
 *
 * Output is sent via send(Object) as an Object[] with:
 *   [0] = combo_key        (String)
 *   [1] = is_anomaly       (String, "true" or "false")
 *   [2] = last_point_score (String)
 *   [3] = threshold        (String)
 *   [4] = model_version    (String)
 *   [5] = window_end       (String, echoed)
 */
@PropertyTemplate(
    name = "FCVAEScoreCaller",
    type = AdapterType.process,
    properties = {
        @PropertyTemplateProperty(name = "apiEndpoint", type = String.class, required = false, defaultValue = "http://localhost:8000/v1/score"),
        @PropertyTemplateProperty(name = "apiKey", type = String.class, required = false, defaultValue = ""),
        @PropertyTemplateProperty(name = "timeoutMs", type = Integer.class, required = false, defaultValue = "5000"),
        @PropertyTemplateProperty(name = "maxRetries", type = Integer.class, required = false, defaultValue = "3")
    },
    outputType = DailyPayloadStream_Type_1_0.class,  // Striim sends typed DailyPayload to us
    inputType = ScorerResult_1_0.class                // We send ScorerResult back to Striim
)
public class FCVAEScoreCaller extends StriimOpenProcessor {

    private static final Logger logger = LogManager.getLogger(FCVAEScoreCaller.class);

    private HttpClient httpClient;

    // Property fields populated by Striim from @PropertyTemplateProperty values
    private String apiEndpoint;
    private String apiKey;
    private int timeoutMs;
    private int maxRetries;

    // No init() method -- StriimOpenProcessor does not have one.
    // We lazily initialize the HttpClient on first run() call instead.

    @Override
    public void run() {
        // Lazy init the HTTP client on first invocation
        if (httpClient == null) {
            int effectiveTimeout = (timeoutMs > 0) ? timeoutMs : 5000;
            int effectiveRetries = (maxRetries > 0) ? maxRetries : 3;
            this.timeoutMs = effectiveTimeout;
            this.maxRetries = effectiveRetries;
            httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(effectiveTimeout))
                .build();
            if (apiEndpoint == null || apiEndpoint.isEmpty()) {
                apiEndpoint = "http://localhost:8000/v1/score";
                logger.warn("apiEndpoint was null/empty, using default: {}", apiEndpoint);
            }
            logger.info("FCVAEScoreCaller initialized. Endpoint: {}, Timeout: {}ms, Retries: {}",
                apiEndpoint, effectiveTimeout, effectiveRetries);
        }

        IBatch<WAEvent> batch = getAdded();
        if (batch == null) return;

        Iterator<WAEvent> it = batch.iterator();
        while (it.hasNext()) {
            WAEvent inputEvent = it.next();

            try {
                // The data field on WAEvent is a generic Object.
                // When coming from a CQ with typed output, it will be
                // the generated type. When coming from a CQ that produces
                // a WAEvent (e.g. from a DSVParser source), data is Object[].
                // We handle it by converting to string representation.
                Object eventData = inputEvent.data;
                String comboKey;
                String valuesJson;
                String windowEnd;

                // If data is itself a WAEvent (possibly a different WAEvent class),
                // unwrap it by accessing its 'data' field via reflection
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
                        // data[2] is window_size (skip), data[3] is window_start (skip)
                        windowEnd = arr.length > 4 ? String.valueOf(arr[4]).trim() : "";
                        valuesJson = normalizeValuesList(rawValues);
                    }else {
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
                // Build the JSON request body for POST /v1/score
                JsonObject body = new JsonObject();
                body.addProperty("combo", comboKey);
                body.add("values", JsonParser.parseString(valuesJson));

                // Call the API
                String responseBody = callApiWithRetry(body.toString());

                if (responseBody == null) {
                    logger.error("All retries exhausted for combo={}, window_end={}",
                        comboKey, windowEnd);
                    continue;
                }

                // Parse the JSON response
                JsonObject resp = JsonParser.parseString(responseBody).getAsJsonObject();

                String isAnomaly = resp.get("is_anomaly").getAsBoolean() ? "true" : "false";
                String lastPointScore = String.valueOf(resp.get("last_point_score").getAsDouble());
                String threshold = String.valueOf(resp.get("threshold").getAsDouble());
                String modelVersion = resp.has("model_version")
                    ? resp.get("model_version").getAsString() : "unknown";

                logger.info("Scored combo={}, is_anomaly={}, score={}, threshold={}",
                    comboKey, isAnomaly, lastPointScore, threshold);

                // Emit result downstream using send(Object)
                // The downstream CQ will receive this as WAEvent.data
                ScorerResult_1_0 result = new ScorerResult_1_0();
                result.combo_key = comboKey;
                result.is_anomaly = String.valueOf(isAnomaly);
                result.anomaly_score = String.valueOf(lastPointScore);
                result.threshold = String.valueOf(threshold);
                result.window_end = windowEnd;

                send(result);

                logger.info("Sent typed result: combo={}, anomaly={}, score={}, threshold={}",
                    comboKey, isAnomaly, lastPointScore, threshold);

            } catch (Exception e) {
                logger.error("Error processing event: {}", e.getMessage(), e);
            }
        }
    }

    private String callApiWithRetry(String jsonBody) {
        int attempt = 0;
        long backoffMs = 500;

        while (attempt <= maxRetries) {
            try {
                HttpRequest.Builder reqBuilder = HttpRequest.newBuilder()
                    .uri(URI.create(apiEndpoint))
                    .timeout(Duration.ofMillis(timeoutMs))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody));

                if (apiKey != null && !apiKey.isEmpty()) {
                    reqBuilder.header("X-Api-Key", apiKey);
                }

                HttpResponse<String> response = httpClient.send(
                    reqBuilder.build(),
                    HttpResponse.BodyHandlers.ofString()
                );

                int status = response.statusCode();

                if (status >= 200 && status < 300) {
                    return response.body();
                }

                if (status >= 400 && status < 500) {
                    logger.error("Client error from FCVAE API: status={}, body={}",
                        status, response.body());
                    return null;
                }

                logger.warn("Server error from FCVAE API: status={}, attempt={}/{}",
                    status, attempt, maxRetries);

            } catch (Exception e) {
                logger.warn("Exception calling FCVAE API: {}, attempt={}/{}",
                    e.getMessage(), attempt, maxRetries);
            }

            try {
                Thread.sleep(backoffMs);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return null;
            }
            backoffMs *= 2;
            attempt++;
        }

        return null;
    }

    private String normalizeValuesList(String raw) {
        if (raw == null || raw.isEmpty()) return "[]";

        String cleaned = raw.trim();

        // Strip surrounding brackets if present
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
        logger.info("FCVAEScoreCaller shutting down.");
    }

    @Override
    public Map getAggVec() { return null; }

    @Override
    public void setAggVec(Map aggVec) { }
}
