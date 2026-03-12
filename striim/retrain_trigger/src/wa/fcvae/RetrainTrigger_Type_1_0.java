package wa.fcvae;

import com.webaction.event.SimpleEvent;
import com.webaction.event.WactionConvertible;
import com.webaction.uuid.UUID;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoSerializable;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.Serializable;
import java.util.Map;

/**
 * Striim user type for the retrain trigger stream.
 *
 * Carries the schedule-window summary that tells the
 * RetrainTriggerCaller OP which combo to retrain.
 *
 * Fields are all Strings (Striim convention for CQ-produced types).
 *
 * Package must be wa.fcvae to match the fcvae namespace used by
 * the existing types (ScorerResult_1_0, RetrainEvent_1_0, etc.).
 *
 * Follows the exact pattern of RetrainEvent_1_0.java and
 * TypedRetrainInput_Type_1_0.java: extends SimpleEvent,
 * implements WactionConvertible + Serializable + KryoSerializable,
 * provides getPayload/setPayload/Kryo/JSON/ContextMap methods.
 */
public class RetrainTrigger_Type_1_0 extends SimpleEvent
        implements WactionConvertible, Serializable, KryoSerializable {

    private static final long serialVersionUID = 1L;
    public static ObjectMapper mapper = newMapper();

    // ---------------------------------------------------------------
    // Fields (populated by the RetrainTrigger CQ)
    // ---------------------------------------------------------------

    /** Combo key, e.g. "Accel_CMP". From GROUP BY combo_key. */
    public String combo;

    /** Retrain mode label, e.g. "weekly". Literal from CQ. */
    public String mode;

    /** Number of scored windows in the schedule window. TO_STRING(COUNT(*)). */
    public String windows_scored;

    /** Timestamp of the last scored window in the schedule window. */
    public String trigger_time;

    // ---------------------------------------------------------------
    // Constructors
    // ---------------------------------------------------------------

    public RetrainTrigger_Type_1_0() { super(); }
    public RetrainTrigger_Type_1_0(long timeStamp) { super(timeStamp); }

    // ---------------------------------------------------------------
    // Getters / Setters (required by Striim introspection)
    // ---------------------------------------------------------------

    public String getCombo() { return combo; }
    public void setCombo(String val) { combo = val; }
    public String getMode() { return mode; }
    public void setMode(String val) { mode = val; }
    public String getWindows_scored() { return windows_scored; }
    public void setWindows_scored(String val) { windows_scored = val; }
    public String getTrigger_time() { return trigger_time; }
    public void setTrigger_time(String val) { trigger_time = val; }

    // ---------------------------------------------------------------
    // Payload (used by Striim for WAEvent.data interop)
    // ---------------------------------------------------------------

    public Object[] getPayload() {
        return new Object[] { combo, mode, windows_scored, trigger_time };
    }

    public void setPayload(Object[] payload) {
        if (payload != null && payload.length >= 4) {
            combo          = (String) payload[0];
            mode           = (String) payload[1];
            windows_scored = (String) payload[2];
            trigger_time   = (String) payload[3];
        }
    }

    // ---------------------------------------------------------------
    // Kryo serialization
    // ---------------------------------------------------------------

    public void write(Kryo kryo, Output output) {
        output.writeString(combo);
        output.writeString(mode);
        output.writeString(windows_scored);
        output.writeString(trigger_time);
    }

    public void read(Kryo kryo, Input input) {
        combo          = input.readString();
        mode           = input.readString();
        windows_scored = input.readString();
        trigger_time   = input.readString();
    }

    // ---------------------------------------------------------------
    // WactionConvertible (context map interop)
    // ---------------------------------------------------------------

    @SuppressWarnings("unchecked")
    public boolean setFromContextMap(Map map) {
        Object ts = map.get("timestamp");
        if (ts instanceof Long) {
            this.setTimeStamp(((Long) ts).longValue());
        }
        Object uid = map.get("uuid");
        if (uid instanceof UUID) {
            this._wa_SimpleEvent_ID = (UUID) uid;
        }
        Object k = map.get("key");
        if (k != null) {
            this.key = k.toString();
        }
        Object v;
        v = map.get("context-combo");
        if (v != null) combo = v.toString();
        v = map.get("context-mode");
        if (v != null) mode = v.toString();
        v = map.get("context-windows_scored");
        if (v != null) windows_scored = v.toString();
        v = map.get("context-trigger_time");
        if (v != null) trigger_time = v.toString();
        return true;
    }

    @SuppressWarnings("unchecked")
    public void convertFromWactionToEvent(
            long timeStamp, UUID id, String key, Map map) {
        this.setTimeStamp(timeStamp);
        if (id != null) this.setID(id);
        if (key != null) this.setKey(key);
        if (map != null) setFromContextMap(map);
    }

    public RetrainTrigger_Type_1_0 convertToDeleteEvent() {
        RetrainTrigger_Type_1_0 del = new RetrainTrigger_Type_1_0();
        del.combo = null;
        del.mode = null;
        del.windows_scored = null;
        del.trigger_time = null;
        return del;
    }

    // ---------------------------------------------------------------
    // JSON (used by JSONFormatter and REST)
    // ---------------------------------------------------------------

    public Object fromJSON(String json) {
        try { return mapper.readValue(json, this.getClass()); }
        catch (Exception e) { return null; }
    }

    public String toJSON() {
        try { return mapper.writeValueAsString(this); }
        catch (Exception e) { return null; }
    }

    // ---------------------------------------------------------------
    // toString
    // ---------------------------------------------------------------

    public String toString() {
        return "RetrainTrigger_Type_1_0{"
            + "combo=" + combo
            + ", mode=" + mode
            + ", windows_scored=" + windows_scored
            + ", trigger_time=" + trigger_time
            + "}";
    }

    // ---------------------------------------------------------------
    // ObjectMapper factory (matches existing types exactly)
    // ---------------------------------------------------------------

    private static ObjectMapper newMapper() {
        try {
            java.lang.reflect.Method m =
                com.webaction.event.ObjectMapperFactory.class
                    .getMethod("getFullInstance");
            return (ObjectMapper) m.invoke(null);
        } catch (Exception e1) {
            try {
                java.lang.reflect.Method m =
                    com.webaction.event.ObjectMapperFactory.class
                        .getMethod("getInstance");
                return (ObjectMapper) m.invoke(null);
            } catch (Exception e2) {
                return new ObjectMapper();
            }
        }
    }
}
