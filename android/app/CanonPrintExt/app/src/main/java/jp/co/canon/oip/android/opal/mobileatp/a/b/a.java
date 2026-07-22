package jp.co.canon.oip.android.opal.mobileatp.a.b;

import java.util.ArrayList;
import jp.co.canon.oip.android.opal.mobileatp.ATPResult;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: ATPAccessToken.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class a implements d {

    /* renamed from: e, reason: collision with root package name */
    public static final String f16865e = "access_token";

    /* renamed from: f, reason: collision with root package name */
    public static final String f16866f = "token_type";

    /* renamed from: g, reason: collision with root package name */
    public static final String f16867g = "expires_in";

    /* renamed from: h, reason: collision with root package name */
    public static final String f16868h = "scope";

    /* renamed from: a, reason: collision with root package name */
    private String f16869a;

    /* renamed from: b, reason: collision with root package name */
    private String f16870b = "";

    /* renamed from: c, reason: collision with root package name */
    private int f16871c = 0;

    /* renamed from: d, reason: collision with root package name */
    private String f16872d = "";

    public a() {
        a();
    }

    public static String a(jp.co.canon.oip.android.opal.mobileatp.util.a aVar) {
        return "7304e5263716973784a387453a";
    }

    public static String b(jp.co.canon.oip.android.opal.mobileatp.util.a aVar, ArrayList<Object> arrayList) {
        return "d4074a9544";
    }

    public void c(String str) {
        this.f16870b = str;
    }

    public String d() {
        return this.f16872d;
    }

    public String e() {
        return this.f16870b;
    }

    public void a(String str) {
        this.f16869a = str;
    }

    public String b() {
        return this.f16869a;
    }

    public int c() {
        return this.f16871c;
    }

    public void a(int i9) {
        this.f16871c = i9;
    }

    public void b(String str) {
        this.f16872d = str;
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.b.d
    public void a(JSONObject jSONObject) {
        a();
        if (jSONObject != null) {
            try {
                this.f16869a = jSONObject.getString("access_token");
                this.f16870b = jSONObject.getString(f16866f);
                this.f16871c = Integer.parseInt(jSONObject.getString(f16867g));
                this.f16872d = jSONObject.getString("scope");
                return;
            } catch (JSONException e10) {
                throw new ATPException(ATPResult.RESULT_CODE_NG_PARSE_JSON, e10.getMessage(), e10);
            }
        }
        throw new ATPException(ATPResult.RESULT_CODE_NG_PARSE_JSON, "JSONObject is null.");
    }

    private void a() {
        this.f16869a = "";
        this.f16870b = "";
        this.f16871c = 0;
        this.f16872d = "";
    }
}
