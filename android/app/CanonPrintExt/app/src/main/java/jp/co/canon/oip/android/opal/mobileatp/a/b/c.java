package jp.co.canon.oip.android.opal.mobileatp.a.b;

import android.util.Base64;
import java.util.ArrayList;
import java.util.Properties;
import jp.co.canon.oip.android.opal.mobileatp.ATPResult;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import jp.co.canon.oip.android.opal.mobileatp.util.g;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: ATPDeviceCredential.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class c implements d {

    /* renamed from: c, reason: collision with root package name */
    public static final String f16875c = "client_id";

    /* renamed from: a, reason: collision with root package name */
    private String f16876a;

    /* renamed from: b, reason: collision with root package name */
    private b f16877b;

    public c() {
        a();
    }

    public static String a(jp.co.canon.oip.android.opal.mobileatp.util.a aVar) {
        return "";
    }

    public static String b(jp.co.canon.oip.android.opal.mobileatp.util.a aVar, ArrayList<Object> arrayList) {
        return "aefc75ed66a88";
    }

    public String c() {
        return this.f16876a;
    }

    public b d() {
        return this.f16877b;
    }

    public Properties e() {
        b bVar;
        if (g.a(this.f16876a) || (bVar = this.f16877b) == null || g.a(bVar.b())) {
            throw new ATPException(106, "client_id or client_secret is empty.");
        }
        Properties properties = new Properties();
        String f10 = jp.co.canon.oip.android.opal.mobileatp.util.c.f(this.f16876a);
        String f11 = jp.co.canon.oip.android.opal.mobileatp.util.c.f(this.f16877b.b());
        properties.setProperty("client_id", f10);
        properties.setProperty(b.f16873b, f11);
        return properties;
    }

    public void a(String str) {
        this.f16876a = str;
    }

    public String b() {
        String b10 = this.f16877b.b();
        if (g.a(this.f16876a) || g.a(b10)) {
            return null;
        }
        StringBuffer stringBuffer = new StringBuffer(this.f16876a);
        stringBuffer.append(':');
        stringBuffer.append(b10);
        return new String(Base64.encode(stringBuffer.toString().getBytes(), 2));
    }

    public void a(b bVar) {
        this.f16877b = bVar;
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.b.d
    public void a(JSONObject jSONObject) {
        a();
        if (jSONObject != null) {
            try {
                this.f16876a = jSONObject.getString("client_id");
                this.f16877b.a(jSONObject);
                return;
            } catch (JSONException e10) {
                throw new ATPException(ATPResult.RESULT_CODE_NG_PARSE_JSON, e10.getMessage(), e10);
            }
        }
        throw new ATPException(ATPResult.RESULT_CODE_NG_PARSE_JSON, "JSONObject is null.");
    }

    public void a(Properties properties) {
        a();
        if (properties != null && properties.size() != 0) {
            if (properties.containsKey("client_id") && properties.containsKey(b.f16873b)) {
                String property = properties.getProperty("client_id");
                String property2 = properties.getProperty(b.f16873b);
                String c10 = jp.co.canon.oip.android.opal.mobileatp.util.c.c(property);
                if (!g.a(c10)) {
                    String c11 = jp.co.canon.oip.android.opal.mobileatp.util.c.c(property2);
                    if (!g.a(c11)) {
                        this.f16876a = c10;
                        this.f16877b.a(c11);
                        return;
                    }
                    throw new ATPException(106, "decryptSecret is empty.");
                }
                throw new ATPException(106, "decryptClientId is empty.");
            }
            throw new ATPException(106, "client_id or client_secret is empty.");
        }
        throw new ATPException(106, "property is empty.");
    }

    private void a() {
        this.f16876a = "";
        this.f16877b = new b();
    }
}
