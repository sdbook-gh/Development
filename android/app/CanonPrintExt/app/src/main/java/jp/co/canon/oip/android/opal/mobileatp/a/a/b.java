package jp.co.canon.oip.android.opal.mobileatp.a.a;

import java.net.URL;
import java.util.ArrayList;
import java.util.Properties;
import jp.co.canon.oip.android.opal.mobileatp.ATPProxySetting;
import jp.co.canon.oip.android.opal.mobileatp.ATPResult;
import jp.co.canon.oip.android.opal.mobileatp.deviceregistration.ATPDeviceRegistrationRequest;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import jp.co.canon.oip.android.opal.mobileatp.util.g;
import org.json.JSONObject;

/* compiled from: ATPCAMSDeviceRegistration.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class b extends a {

    /* renamed from: i, reason: collision with root package name */
    private static final String f16835i = "/auth/oauth2/clients";

    /* renamed from: e, reason: collision with root package name */
    private final String f16836e;

    /* renamed from: f, reason: collision with root package name */
    private final ATPDeviceRegistrationRequest f16837f;

    /* renamed from: g, reason: collision with root package name */
    private final String f16838g;

    /* renamed from: h, reason: collision with root package name */
    private final String f16839h;

    public b(String str, ATPDeviceRegistrationRequest aTPDeviceRegistrationRequest, String str2, String str3, ATPProxySetting aTPProxySetting) {
        this.f16836e = str;
        this.f16837f = aTPDeviceRegistrationRequest;
        this.f16839h = str3;
        this.f16838g = str2;
        this.f16829a = aTPProxySetting;
    }

    public static String a(jp.co.canon.oip.android.opal.mobileatp.util.a aVar) {
        return "5856634d3833645a694e6771e";
    }

    public static String b(jp.co.canon.oip.android.opal.mobileatp.util.a aVar, ArrayList<Object> arrayList) {
        return "207c0a";
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.a.a
    public String c() {
        return this.f16839h;
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.a.a
    public String d() {
        return new jp.co.canon.oip.android.opal.mobileatp.a.c.a(this.f16836e, this.f16837f).a();
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.a.a
    public Properties e() {
        Properties properties = new Properties();
        String n9 = n();
        if (g.a(n9)) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_CREATE_REQUEST_MESSAGE, "Content-Type is empty.");
        }
        properties.setProperty("Content-Type", n9);
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "Content-Type = " + n9);
        return properties;
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.a.a
    public String f() {
        return f16835i;
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.a.a
    public String g() {
        return this.f16838g;
    }

    public String n() {
        return e.f16851h;
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.a.a
    public jp.co.canon.oip.android.opal.mobileatp.a.b.d a(JSONObject jSONObject) {
        jp.co.canon.oip.android.opal.mobileatp.a.b.c cVar = new jp.co.canon.oip.android.opal.mobileatp.a.b.c();
        cVar.a(jSONObject);
        return cVar;
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.a.a
    public URL b() {
        return a(this.f16830b.getRegistrationServerName(), this.f16837f.getRealm());
    }
}
