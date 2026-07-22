package jp.co.canon.oip.android.opal.mobileatp.a.b;

import java.util.ArrayList;
import jp.co.canon.oip.android.opal.mobileatp.ATPResult;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: ATPClientSecret.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class b implements d {

    /* renamed from: b, reason: collision with root package name */
    public static final String f16873b = "client_secret";

    /* renamed from: a, reason: collision with root package name */
    private String f16874a;

    public b() {
        a();
    }

    public static String a(jp.co.canon.oip.android.opal.mobileatp.util.a aVar) {
        return "53dfabbd2d6307640";
    }

    public static String b(jp.co.canon.oip.android.opal.mobileatp.util.a aVar, ArrayList<Object> arrayList) {
        return "55e3ac9f884363";
    }

    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        try {
            return this.f16874a.equals(((b) obj).b());
        } catch (Exception unused) {
            return false;
        }
    }

    public int hashCode() {
        return super.hashCode();
    }

    public void a(String str) {
        this.f16874a = str;
    }

    public String b() {
        return this.f16874a;
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.b.d
    public void a(JSONObject jSONObject) {
        a();
        if (jSONObject != null) {
            try {
                this.f16874a = jSONObject.getString(f16873b);
                return;
            } catch (JSONException e10) {
                throw new ATPException(ATPResult.RESULT_CODE_NG_PARSE_JSON, e10.getMessage(), e10);
            }
        }
        throw new ATPException(ATPResult.RESULT_CODE_NG_PARSE_JSON, "JSONObject is null.");
    }

    private void a() {
        this.f16874a = "";
    }
}
