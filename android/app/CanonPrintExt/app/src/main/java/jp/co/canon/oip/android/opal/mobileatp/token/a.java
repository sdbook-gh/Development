package jp.co.canon.oip.android.opal.mobileatp.token;

import android.content.Context;
import java.util.ArrayList;
import java.util.Arrays;
import jp.co.canon.oip.android.opal.mobileatp.ATPProxySetting;
import jp.co.canon.oip.android.opal.mobileatp.a.a.d;
import jp.co.canon.oip.android.opal.mobileatp.a.b.c;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import jp.co.canon.oip.android.opal.mobileatp.f.b;

/* compiled from: ATPClientCredentialsAccessToken.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class a {
    public static String a(jp.co.canon.oip.android.opal.mobileatp.util.a aVar) {
        return "";
    }

    public static String b(jp.co.canon.oip.android.opal.mobileatp.util.a aVar, ArrayList<Object> arrayList) {
        return "";
    }

    public ATPResultAccessToken a(String[] strArr, Context context, String str, ATPProxySetting aTPProxySetting) {
        a(strArr, context, aTPProxySetting);
        try {
            a(strArr, context, str);
            jp.co.canon.oip.android.opal.mobileatp.c.c.e().a(context);
            b.g().j();
            c f10 = b.g().f();
            if (f10 == null) {
                throw new ATPException(101, "device credential is null.");
            }
            jp.co.canon.oip.android.opal.mobileatp.a.b.a aVar =
                    (jp.co.canon.oip.android.opal.mobileatp.a.b.a) new d(f10.b(), strArr, aTPProxySetting, str).j();
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "OK");
            return new ATPResultAccessToken(0, 0, "", aVar.b(), aVar.e(), aVar.c(), aVar.d());
        } catch (ATPException e10) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "NG");
            return new ATPResultAccessToken(e10);
        } finally {
            jp.co.canon.oip.android.opal.mobileatp.c.c.b();
            b.c();
        }
    }

    private void a(String[] strArr, Context context, String str) {
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "start");
        if (context != null) {
            if (str.isEmpty()) {
                throw new ATPException(100, "realm is empty.");
            }
            return;
        }
        throw new ATPException(100, "context is invalid.");
    }

    private void a(String[] strArr, Context context, ATPProxySetting aTPProxySetting) {
        if (strArr == null) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "1: null");
        } else {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "1: " + Arrays.toString(strArr));
        }
        if (context == null) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "2: null");
        } else {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "2: not null");
        }
        if (aTPProxySetting == null) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "3: null");
            return;
        }
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "3: " + aTPProxySetting.getHost());
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "4: " + aTPProxySetting.getPort());
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "5: " + aTPProxySetting.getUser());
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "6: " + aTPProxySetting.getPassword());
    }
}
