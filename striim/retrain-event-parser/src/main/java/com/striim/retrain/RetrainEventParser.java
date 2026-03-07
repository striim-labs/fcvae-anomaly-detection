package com.striim.retrain;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.webaction.anno.AdapterType;
import com.webaction.anno.PropertyTemplate;
import com.webaction.anno.PropertyTemplateProperty;
import com.webaction.runtime.components.openprocessor.StriimOpenProcessor;
import com.webaction.runtime.containers.IBatch;
import com.webaction.runtime.containers.WAEvent;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import wa.fcvae.RetrainEvent_1_0;
import wa.fcvae.TypedRetrainInput_Type_1_0;

import java.lang.reflect.Field;
import java.util.Iterator;
import java.util.Map;

/**
 * Striim Open Processor that parses JSONL lines from model_events.py
 * into typed RetrainEvent fields.
 *
 * Input stream:  TypedRetrainInput OF fcvae.RawJsonLine
 *                (CQ converts FileReader WAEvent -> RawJsonLine with raw_line field)
 * Output stream: RetrainEventStream OF fcvae.RetrainEvent
 *
 * Annotation naming follows Striim convention:
 *   outputType = what Striim sends TO this OP  = TypedRetrainInput_Type_1_0
 *   inputType  = what this OP sends BACK        = RetrainEvent_1_0
 */
@PropertyTemplate(
    name = "RetrainEventParser",
    type = AdapterType.process,
    properties = {
        @PropertyTemplateProperty(
            name = "placeholder",
            type = String.class,
            required = false,
            defaultValue = "none"
        )
    },
    outputType = TypedRetrainInput_Type_1_0.class,
    inputType  = RetrainEvent_1_0.class
)
public class RetrainEventParser extends StriimOpenProcessor {

    private static final Logger logger = LogManager.getLogger(RetrainEventParser.class);

    // ---------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------

    public RetrainEventParser() {
        logger.info("RetrainEventParser instantiated.");
    }

    // ---------------------------------------------------------------
    // run() -- main entry point called by Striim for each batch.
    // ---------------------------------------------------------------

    @SuppressWarnings({"unchecked", "rawtypes"})
    public void run() {
        try {
            IBatch batch = getAdded();
            if (batch == null) {
                return;
            }

            Iterator it = batch.iterator();
            while (it.hasNext()) {
                Object inputEvent = it.next();
                try {
                    processEvent(inputEvent);
                } catch (Exception e) {
                    logger.error("Error processing event: {}", e.getMessage(), e);
                }
            }
        } catch (Exception e) {
            logger.error("Error in run(): {}", e.getMessage(), e);
        }
    }

    // ---------------------------------------------------------------
    // processEvent() -- handles a single event from the batch.
    //
    // The input arrives as a WAEvent wrapper. Based on the working
    // FCVAEScoreCaller pattern, the typed data is in WAEvent.data
    // (cast to the stream type class).
    // ---------------------------------------------------------------

    private void processEvent(Object inputEvent) throws Exception {
        String rawJson = null;

        // --- Extract the raw JSON line from the input event ---
        // At runtime, Striim wraps typed events inside WAEvent.data.
        // Following the FCVAEScoreCaller pattern:
        //   SendToOPStream_Type_1_0 type = (SendToOPStream_Type_1_0) it.next().data;

        Object dataHolder = inputEvent;

        // Unwrap WAEvent wrapper(s) to get to .data
        if (dataHolder instanceof WAEvent) {
            Object inner = ((WAEvent) dataHolder).data;
            if (inner instanceof TypedRetrainInput_Type_1_0) {
                rawJson = ((TypedRetrainInput_Type_1_0) inner).raw_line;
            } else if (inner instanceof Object[]) {
                // Fallback: data is Object[] from DSVParser
                Object[] arr = (Object[]) inner;
                if (arr.length > 0 && arr[0] != null) {
                    rawJson = String.valueOf(arr[0]).trim();
                }
            } else if (inner != null) {
                // Try reflection for runtime WAEvent class mismatch
                rawJson = extractRawLineViaReflection(inner);
            }
        } else if (dataHolder instanceof TypedRetrainInput_Type_1_0) {
            rawJson = ((TypedRetrainInput_Type_1_0) dataHolder).raw_line;
        } else {
            // Reflection fallback for the dual-WAEvent class issue
            rawJson = extractFromUnknownEvent(dataHolder);
        }

        if (rawJson == null || rawJson.trim().isEmpty()) {
            logger.warn("No raw JSON found in event, skipping. Event class: {}",
                    inputEvent.getClass().getName());
            return;
        }
        rawJson = rawJson.trim();

        // --- Parse the JSON ---
        JsonObject json = JsonParser.parseString(rawJson).getAsJsonObject();

        String eventType = safeGetString(json, "event_type", "");
        String eventTime = safeGetString(json, "timestamp", "");
        String comboVal  = safeGetString(json, "combo", "");

        // job_id is nested: $.details.job_id
        String jobId = "";
        if (json.has("details") && json.get("details").isJsonObject()) {
            JsonObject details = json.getAsJsonObject("details");
            jobId = safeGetString(details, "job_id", "");
        }

        // event_details = full raw JSON line
        String eventDetails = rawJson;

        // --- Build typed output and send ---
        RetrainEvent_1_0 result = new RetrainEvent_1_0();
        result.event_type = eventType;
        result.event_time = eventTime;
        result.combo = comboVal;
        result.job_id = jobId;
        result.event_details = eventDetails;

        logger.info("Parsed event: type={}, time={}, combo={}, job_id={}",
                eventType, eventTime, comboVal, jobId);

        send(result);
    }

    // ---------------------------------------------------------------
    // close() -- cleanup
    // ---------------------------------------------------------------

    public void close() throws Exception {
        logger.info("RetrainEventParser shutting down.");
    }

    // ---------------------------------------------------------------
    // Required by Processor interface (raw Map -- no generics)
    // ---------------------------------------------------------------

    @SuppressWarnings("rawtypes")
    public Map getAggVec() {
        return null;
    }

    @SuppressWarnings("rawtypes")
    public void setAggVec(Map aggVec) {
        // no-op
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    /**
     * Try to extract raw_line field via reflection from a typed event
     * whose class may differ at runtime.
     */
    private String extractRawLineViaReflection(Object obj) {
        try {
            Field f = obj.getClass().getField("raw_line");
            Object val = f.get(obj);
            return val != null ? val.toString().trim() : null;
        } catch (NoSuchFieldException e) {
            // Not a RawJsonLine type, try data[0] fallback
            return extractDataArrayViaReflection(obj);
        } catch (IllegalAccessException e) {
            logger.error("Cannot access raw_line field: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Fallback: extract data[0] from an unknown WAEvent-like object.
     */
    private String extractFromUnknownEvent(Object obj) {
        // First check if it has a raw_line field (typed input)
        String result = extractRawLineViaReflection(obj);
        if (result != null) return result;

        // Then try WAEvent-style .data field
        try {
            Field dataField = obj.getClass().getField("data");
            Object dataObj = dataField.get(obj);

            // data might itself be a typed event
            if (dataObj != null) {
                result = extractRawLineViaReflection(dataObj);
                if (result != null) return result;

                // Or data might be Object[]
                if (dataObj instanceof Object[]) {
                    Object[] arr = (Object[]) dataObj;
                    if (arr.length > 0 && arr[0] != null) {
                        return String.valueOf(arr[0]).trim();
                    }
                }
            }
        } catch (NoSuchFieldException | IllegalAccessException e) {
            logger.warn("Cannot extract data from event of type {}: {}",
                    obj.getClass().getName(), e.getMessage());
        }
        return null;
    }

    /**
     * Extract data[0] from an object with an Object[] data field.
     */
    private String extractDataArrayViaReflection(Object obj) {
        try {
            Field dataField = obj.getClass().getField("data");
            Object dataObj = dataField.get(obj);
            if (dataObj instanceof Object[]) {
                Object[] arr = (Object[]) dataObj;
                if (arr.length > 0 && arr[0] != null) {
                    return String.valueOf(arr[0]).trim();
                }
            }
        } catch (NoSuchFieldException | IllegalAccessException e) {
            // Not a WAEvent-like object
        }
        return null;
    }

    private static String safeGetString(JsonObject json, String key, String defaultValue) {
        if (json.has(key)) {
            JsonElement el = json.get(key);
            if (el.isJsonNull()) {
                return defaultValue;
            }
            return el.getAsString();
        }
        return defaultValue;
    }
}