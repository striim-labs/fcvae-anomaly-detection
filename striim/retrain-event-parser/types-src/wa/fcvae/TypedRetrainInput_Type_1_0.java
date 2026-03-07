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

public class TypedRetrainInput_Type_1_0 extends SimpleEvent
        implements WactionConvertible, Serializable, KryoSerializable {

    private static final long serialVersionUID = 1L;
    public static ObjectMapper mapper = newMapper();

    public String raw_line;

    public TypedRetrainInput_Type_1_0() { super(); }
    public TypedRetrainInput_Type_1_0(long timeStamp) { super(timeStamp); }

    public String getRaw_line() { return raw_line; }
    public void setRaw_line(String val) { raw_line = val; }

    public Object[] getPayload() {
        return new Object[] { raw_line };
    }

    public void setPayload(Object[] payload) {
        if (payload != null && payload.length >= 1) {
            raw_line = (String) payload[0];
        }
    }

    public void write(Kryo kryo, Output output) {
        output.writeString(raw_line);
    }

    public void read(Kryo kryo, Input input) {
        raw_line = input.readString();
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
        v = map.get("context-raw_line"); if (v != null) raw_line = v.toString();
        return true;
    }

    public void convertFromWactionToEvent(long timeStamp, UUID id, String key, Map map) {
        this.setTimeStamp(timeStamp);
        if (id != null) this.setID(id);
        if (key != null) this.setKey(key);
        if (map != null) setFromContextMap(map);
    }

    public TypedRetrainInput_Type_1_0 convertToDeleteEvent() {
        TypedRetrainInput_Type_1_0 del = new TypedRetrainInput_Type_1_0();
        del.raw_line = null;
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
        return "TypedRetrainInput_Type_1_0{raw_line=" + raw_line + "}";
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