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

public class RetrainEvent_1_0 extends SimpleEvent
        implements WactionConvertible, Serializable, KryoSerializable {

    private static final long serialVersionUID = 1L;
    public static ObjectMapper mapper = newMapper();

    public String event_type;
    public String event_time;
    public String combo;
    public String job_id;
    public String event_details;

    public RetrainEvent_1_0() { super(); }
    public RetrainEvent_1_0(long timeStamp) { super(timeStamp); }

    public String getEvent_type() { return event_type; }
    public void setEvent_type(String val) { event_type = val; }
    public String getEvent_time() { return event_time; }
    public void setEvent_time(String val) { event_time = val; }
    public String getCombo() { return combo; }
    public void setCombo(String val) { combo = val; }
    public String getJob_id() { return job_id; }
    public void setJob_id(String val) { job_id = val; }
    public String getEvent_details() { return event_details; }
    public void setEvent_details(String val) { event_details = val; }

    public Object[] getPayload() {
        return new Object[] { event_type, event_time, combo, job_id, event_details };
    }

    public void setPayload(Object[] payload) {
        if (payload != null && payload.length >= 5) {
            event_type    = (String) payload[0];
            event_time    = (String) payload[1];
            combo         = (String) payload[2];
            job_id        = (String) payload[3];
            event_details = (String) payload[4];
        }
    }

    public void write(Kryo kryo, Output output) {
        output.writeString(event_type);
        output.writeString(event_time);
        output.writeString(combo);
        output.writeString(job_id);
        output.writeString(event_details);
    }

    public void read(Kryo kryo, Input input) {
        event_type    = input.readString();
        event_time    = input.readString();
        combo         = input.readString();
        job_id        = input.readString();
        event_details = input.readString();
    }

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
        v = map.get("context-event_type");    if (v != null) event_type    = v.toString();
        v = map.get("context-event_time");    if (v != null) event_time    = v.toString();
        v = map.get("context-combo");         if (v != null) combo         = v.toString();
        v = map.get("context-job_id");        if (v != null) job_id        = v.toString();
        v = map.get("context-event_details"); if (v != null) event_details = v.toString();
        return true;
    }

    public void convertFromWactionToEvent(long timeStamp, UUID id, String key, Map map) {
        this.setTimeStamp(timeStamp);
        if (id != null) this.setID(id);
        if (key != null) this.setKey(key);
        if (map != null) setFromContextMap(map);
    }

    public RetrainEvent_1_0 convertToDeleteEvent() {
        RetrainEvent_1_0 del = new RetrainEvent_1_0();
        del.event_type = null;
        del.event_time = null;
        del.combo = null;
        del.job_id = null;
        del.event_details = null;
        return del;
    }

    public Object fromJSON(String json) {
        try { return mapper.readValue(json, this.getClass()); }
        catch (Exception e) { return null; }
    }

    public String toJSON() {
        try { return mapper.writeValueAsString(this); }
        catch (Exception e) { return null; }
    }

    public String toString() {
        return "RetrainEvent_1_0{"
            + "event_type=" + event_type
            + ", event_time=" + event_time
            + ", combo=" + combo
            + ", job_id=" + job_id
            + ", event_details=" + event_details
            + "}";
    }

    private static ObjectMapper newMapper() {
        try {
            java.lang.reflect.Method m =
                com.webaction.event.ObjectMapperFactory.class.getMethod("getFullInstance");
            return (ObjectMapper) m.invoke(null);
        } catch (Exception e1) {
            try {
                java.lang.reflect.Method m =
                    com.webaction.event.ObjectMapperFactory.class.getMethod("getInstance");
                return (ObjectMapper) m.invoke(null);
            } catch (Exception e2) {
                return new ObjectMapper();
            }
        }
    }
}