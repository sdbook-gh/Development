package jp.co.canon.oip.android.opal.mobileatp.a.c;

import java.util.ArrayList;
import jp.co.canon.oip.android.opal.mobileatp.ATPResult;
import jp.co.canon.oip.android.opal.mobileatp.a.a.e;
import jp.co.canon.oip.android.opal.mobileatp.d.b;
import jp.co.canon.oip.android.opal.mobileatp.deviceregistration.ATPDeviceRegistrationRequest;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: ATPJSONDeviceRegistrationRequest.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class a {

    /* renamed from: a, reason: collision with root package name */
    String f16878a;

    /* renamed from: b, reason: collision with root package name */
    ATPDeviceRegistrationRequest f16879b;

    public a(String str, ATPDeviceRegistrationRequest aTPDeviceRegistrationRequest) {
        this.f16878a = str;
        this.f16879b = aTPDeviceRegistrationRequest;
    }

    public static String a(jp.co.canon.oip.android.opal.mobileatp.util.a aVar) {
        return "";
    }

    public static String b(jp.co.canon.oip.android.opal.mobileatp.util.a aVar, ArrayList<Object> arrayList) {
        return "";
    }

    public String a() {
        JSONObject jSONObject = new JSONObject();
        try {
            a(jSONObject, "device_id", this.f16878a);
            a(jSONObject, e.f16859p, this.f16879b.getApplicationId());
            a(jSONObject, "client_name", this.f16879b.getClientName());
            a(jSONObject, "client_description", this.f16879b.getClientDescription());
            a(jSONObject, "scopes", this.f16879b.getScopes());
            a(jSONObject, "default_scopes", this.f16879b.getDefaultScopes());
            b.a(3, jSONObject.toString());
            return jSONObject.toString();
        } catch (JSONException e10) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_ENCODE_JSON, e10.getMessage(), e10);
        }
    }

    private static void a(JSONObject jSONObject, String str, String str2) throws JSONException {
        if (str2 == null || str2.isEmpty()) {
            return;
        }
        jSONObject.put(str, str2);
    }

    private static void a(JSONObject jSONObject, String str, String[] strArr) throws JSONException {
        if (strArr == null || strArr.length <= 0) {
            return;
        }
        JSONArray jSONArray = new JSONArray();
        for (String str2 : strArr) {
            jSONArray.put(str2);
        }
        jSONObject.put(str, jSONArray);
    }
}
