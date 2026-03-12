package com.striim.fcvae;

import com.webaction.anno.AdapterType;
import com.webaction.anno.PropertyTemplate;
import com.webaction.anno.PropertyTemplateProperty;
import com.webaction.runtime.components.openprocessor.StriimOpenProcessor;
import com.webaction.runtime.containers.WAEvent;
import com.webaction.runtime.containers.IBatch;
import wa.fcvae.RetrainTrigger_Type_1_0;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Iterator;
import java.util.Map;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonArray;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Striim Open Processor that triggers FCVAE model retraining.
 *
 * Consumes RetrainTrigger_Type_1_0 events from a schedule-based
 * CQ that fires when a jumping window (e.g., 7 DAYS) closes.
 * Each event carries a single combo_key. The OP calls
 * POST /v1/retrain with auto_reload=true and logs the result.
 *
 * This is a fire-and-forget OP: it never calls send().
 * The FastAPI retrain service handles the full lifecycle
 * (train, validate, promote, hot-swap, event logging).
 *
 * Annotation convention (same as FCVAEScoreCaller):
 *   outputType = what Striim sends TO this OP
 *   inputType  = what this OP sends BACK to Striim
 *
 * Since this OP never sends, inputType is set to the same
 * type as outputType to satisfy the annotation requirement.
 */
@PropertyTemplate(
    name = "RetrainTriggerCaller",
    type = AdapterType.process,
    properties = {
        @PropertyTemplateProperty(
            name = "apiEndpoint",
            type = String.class,
            required = false,
            defaultValue = "http://localhost:8000/v1/retrain"),
        @PropertyTemplateProperty(
            name = "apiKey",
            type = String.class,
            required = false,
            defaultValue = ""),
        @PropertyTemplateProperty(
            name = "timeoutMs",
            type = Integer.class,
            required = false,
            defaultValue = "180000"),
        @PropertyTemplateProperty(
            name = "maxRetries",
            type = Integer.class,
            required = false,
            defaultValue = "3"),
        @PropertyTemplateProperty(
            name = "lookbackDays",
            type = Integer.class,
            required = false,
            defaultValue = "9999"),
        @PropertyTemplateProperty(
            name = "minWindows",
            type = Integer.class,
            required = false,
            defaultValue = "50"),
        @PropertyTemplateProperty(
            name = "retrainMode",
            type = String.class,
            required = false,
            defaultValue = "weekly")
    },
    outputType = RetrainTrigger_Type_1_0.class,
    inputType  = RetrainTrigger_Type_1_0.class
)
public class RetrainTriggerCaller extends StriimOpenProcessor {

    private static final Logger logger =
        LogManager.getLogger(RetrainTriggerCaller.class);

    private HttpClient httpClient;

    // Property fields populated by Striim from @PropertyTemplateProperty
    private String apiEndpoint;
    private String apiKey;
    private int timeoutMs;
    private int maxRetries;
    private int lookbackDays;
    private int minWindows;
    private String retrainMode;

    // ---------------------------------------------------------------
    // run() -- main entry point called by Striim for each batch
    // ---------------------------------------------------------------

    @Override
    public void run() {
        // Lazy init the HTTP client on first invocation
        if (httpClient == null) {
            timeoutMs    = (timeoutMs > 0) ? timeoutMs : 180000;
            maxRetries   = (maxRetries > 0) ? maxRetries : 3;
            lookbackDays = (lookbackDays > 0) ? lookbackDays : 9999;
            minWindows   = (minWindows > 0) ? minWindows : 50;
            if (retrainMode == null || retrainMode.isEmpty()) {
                retrainMode = "weekly";
            }
            if (apiEndpoint == null || apiEndpoint.isEmpty()) {
                apiEndpoint = "http://localhost:8000/v1/retrain";
                logger.warn("apiEndpoint was null/empty, using default: {}",
                    apiEndpoint);
            }

            httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(timeoutMs))
                .build();

            logger.info("RetrainTriggerCaller initialized. "
                + "Endpoint: {}, Timeout: {}ms, Retries: {}, "
                + "Mode: {}, LookbackDays: {}, MinWindows: {}",
                apiEndpoint, timeoutMs, maxRetries,
                retrainMode, lookbackDays, minWindows);
        }

        IBatch<WAEvent> batch = getAdded();
        if (batch == null) return;

        Iterator<WAEvent> it = batch.iterator();
        while (it.hasNext()) {
            WAEvent inputEvent = it.next();
            try {
                processEvent(inputEvent);
            } catch (Exception e) {
                logger.error("Error processing retrain trigger event: {}",
                    e.getMessage(), e);
            }
        }
    }

    // ---------------------------------------------------------------
    // processEvent -- extract fields, call retrain API, log result
    // ---------------------------------------------------------------

    private void processEvent(WAEvent inputEvent) {
        Object eventData = inputEvent.data;
        String combo = null;
        String mode = null;
        String windowsScored = null;
        String triggerTime = null;

        // --- Extract fields from the typed event ---
        // Primary path: typed RetrainTrigger_Type_1_0
        if (eventData instanceof RetrainTrigger_Type_1_0) {
            RetrainTrigger_Type_1_0 trigger =
                (RetrainTrigger_Type_1_0) eventData;
            combo         = trigger.combo;
            mode          = trigger.mode;
            windowsScored = trigger.windows_scored;
            triggerTime   = trigger.trigger_time;

        // Fallback: Object[] from untyped CQ
        } else if (eventData instanceof Object[]) {
            Object[] arr = (Object[]) eventData;
            combo         = arr.length > 0
                ? String.valueOf(arr[0]).trim() : "";
            mode          = arr.length > 1
                ? String.valueOf(arr[1]).trim() : retrainMode;
            windowsScored = arr.length > 2
                ? String.valueOf(arr[2]).trim() : "0";
            triggerTime   = arr.length > 3
                ? String.valueOf(arr[3]).trim() : "";

        // Fallback: reflection for runtime class mismatch
        } else if (eventData != null) {
            try {
                combo = String.valueOf(
                    eventData.getClass().getField("combo")
                        .get(eventData)).trim();
                mode = String.valueOf(
                    eventData.getClass().getField("mode")
                        .get(eventData)).trim();
                windowsScored = String.valueOf(
                    eventData.getClass().getField("windows_scored")
                        .get(eventData)).trim();
                triggerTime = String.valueOf(
                    eventData.getClass().getField("trigger_time")
                        .get(eventData)).trim();
            } catch (Exception e) {
                logger.error(
                    "Cannot extract fields from event of type: {}",
                    eventData.getClass().getName());
                return;
            }
        } else {
            logger.warn("Null event data in retrain trigger, skipping");
            return;
        }

        // --- Validate combo ---
        if (combo == null || combo.isEmpty()) {
            logger.warn("Empty combo in retrain trigger event, skipping");
            return;
        }

        logger.info("Retrain trigger fired: combo={}, mode={}, "
            + "windows_scored={}, trigger_time={}",
            combo, mode, windowsScored, triggerTime);

        // --- Build JSON body for POST /v1/retrain ---
        JsonObject body = new JsonObject();
        JsonArray combos = new JsonArray();
        combos.add(combo);
        body.add("combos", combos);
        body.addProperty("mode",
            (mode != null && !mode.isEmpty()) ? mode : retrainMode);
        body.addProperty("auto_reload", true);
        body.addProperty("lookback_days", lookbackDays);
        body.addProperty("min_windows", minWindows);

        // --- Call the retrain API ---
        String responseBody = callApiWithRetry(body.toString());

        if (responseBody == null) {
            logger.error(
                "Retrain trigger FAILED for combo={} "
                + "(all retries exhausted)", combo);
            return;
        }

        // --- Parse the 202 response and log ---
        try {
            JsonObject resp = JsonParser.parseString(responseBody)
                .getAsJsonObject();
            String jobId = resp.has("job_id")
                ? resp.get("job_id").getAsString() : "unknown";
            String status = resp.has("status")
                ? resp.get("status").getAsString() : "unknown";

            logger.info(
                "Retrain triggered successfully: "
                + "combo={}, job_id={}, status={}",
                combo, jobId, status);
        } catch (Exception e) {
            logger.warn(
                "Could not parse retrain response for combo={}: {}",
                combo, responseBody);
        }

        // Fire-and-forget: no send() call.
        // The retrain runs asynchronously in the FastAPI service.
        // Lifecycle events will appear in model_events.jsonl and
        // be picked up by the RetrainMonitor pipeline.
    }

    // ---------------------------------------------------------------
    // HTTP retry logic (same pattern as FCVAEScoreCaller)
    // ---------------------------------------------------------------

    private String callApiWithRetry(String jsonBody) {
        int attempt = 0;
        long backoffMs = 1000;  // 1s initial (longer than scoring's 500ms)

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

                // 202 Accepted is the expected success response
                if (status >= 200 && status < 300) {
                    return response.body();
                }

                // 4xx = client error, do not retry
                // This includes 409 Conflict (retrain already running)
                if (status >= 400 && status < 500) {
                    logger.error(
                        "Client error from retrain API: status={}, body={}",
                        status, response.body());
                    return null;
                }

                // 5xx = server error, retry
                logger.warn(
                    "Server error from retrain API: status={}, attempt={}/{}",
                    status, attempt, maxRetries);

            } catch (Exception e) {
                logger.warn(
                    "Exception calling retrain API: {}, attempt={}/{}",
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

    // ---------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------

    @Override
    public void close() throws Exception {
        logger.info("RetrainTriggerCaller shutting down.");
    }

    @Override
    public Map getAggVec() { return null; }

    @Override
    public void setAggVec(Map aggVec) { }
}
