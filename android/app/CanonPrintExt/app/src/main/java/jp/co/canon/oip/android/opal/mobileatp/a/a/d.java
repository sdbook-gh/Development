package jp.co.canon.oip.android.opal.mobileatp.a.a;

import java.io.UnsupportedEncodingException;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;
import jp.co.canon.android.cnml.common.CNMLJCmnUtil;
import jp.co.canon.oip.android.opal.mobileatp.ATPProxySetting;
import jp.co.canon.oip.android.opal.mobileatp.ATPResult;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import jp.co.canon.oip.android.opal.mobileatp.util.g;
import org.json.JSONObject;

/* compiled from: ATPCAMSTokenClientCredentials.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class d extends a {

    /* renamed from: h, reason: collision with root package name */
    private static final String f16840h = "/auth/oauth2/access_token";

    /* renamed from: e, reason: collision with root package name */
    private final String f16841e;

    /* renamed from: f, reason: collision with root package name */
    private final String[] f16842f;

    /* renamed from: g, reason: collision with root package name */
    private final String f16843g;

    public d(String str, String[] strArr, ATPProxySetting aTPProxySetting, String str2) {
        if (g.a(str)) {
            throw new ATPException(103, "authorization is empty.");
        }
        this.f16841e = str;
        this.f16842f = strArr;
        this.f16829a = aTPProxySetting;
        this.f16843g = str2;
    }

    public static String a(jp.co.canon.oip.android.opal.mobileatp.util.a aVar) {
        return "8343578457a32446e30706e72f";
    }

    public static String b(jp.co.canon.oip.android.opal.mobileatp.util.a aVar, ArrayList<Object> arrayList) {
        return "572fe42";
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.a.a
    public String d() {
        StringBuilder sb = new StringBuilder("grant_type=client_credentials&scope=");
        try {
            String[] strArr = this.f16842f;
            String encode = strArr != null ? URLEncoder.encode(a(strArr, CNMLJCmnUtil.STRING_SPACE), "UTF-8") : "";
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, e.f16856m);
            sb.append(encode);
            return sb.toString();
        } catch (UnsupportedEncodingException e10) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_CREATE_REQUEST_URLENCODE, e10.getMessage(), e10);
        }
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
        StringBuffer stringBuffer = new StringBuffer("Basic ");
        stringBuffer.append(this.f16841e);
        properties.setProperty("Authorization", stringBuffer.toString());
        return properties;
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.a.a
    public String f() {
        return f16840h;
    }

    public String n() {
        return e.f16852i;
    }

    private static String a(String[] strArr, String str) {
        StringBuilder sb = new StringBuilder();
        for (String str2 : strArr) {
            if (sb.length() > 0) {
                sb.append(str);
            }
            sb.append(str2);
        }
        return sb.toString();
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.a.a
    public URL b() {
        return a(this.f16830b.getTokenServerName(), this.f16843g);
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.a.a.a
    public jp.co.canon.oip.android.opal.mobileatp.a.b.d a(JSONObject jSONObject) {
        jp.co.canon.oip.android.opal.mobileatp.a.b.a aVar = new jp.co.canon.oip.android.opal.mobileatp.a.b.a();
        aVar.a(jSONObject);
        return aVar;
    }
}
