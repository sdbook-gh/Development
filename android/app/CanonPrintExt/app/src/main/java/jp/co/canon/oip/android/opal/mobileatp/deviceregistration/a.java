package jp.co.canon.oip.android.opal.mobileatp.deviceregistration;

import android.content.Context;
import java.util.regex.Pattern;
import jp.co.canon.android.cnml.common.CNMLJCmnUtil;
import jp.co.canon.oip.android.opal.mobileatp.ATPProxySetting;
import jp.co.canon.oip.android.opal.mobileatp.ATPResult;
import jp.co.canon.oip.android.opal.mobileatp.c.c;
import jp.co.canon.oip.android.opal.mobileatp.d.b;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import jp.co.canon.oip.android.opal.mobileatp.util.g;

/* compiled from: ATPDeviceRegistration.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class a {
    private void a(ATPDeviceRegistrationRequest aTPDeviceRegistrationRequest, Context context) {
        b.a(3, "start");
        if (aTPDeviceRegistrationRequest == null) {
            throw new ATPException(100, "deviceRegistrationRequest is invalid.");
        }
        if (context == null) {
            throw new ATPException(100, "context is invalid.");
        }
        if (aTPDeviceRegistrationRequest.getRealm().isEmpty()) {
            throw new ATPException(100, "realm is empty.");
        }
    }

    public ATPResult b(ATPDeviceRegistrationRequest aTPDeviceRegistrationRequest, Context context, ATPProxySetting aTPProxySetting) {
        a(aTPDeviceRegistrationRequest, context, aTPProxySetting);
        try {
            try {
                a(aTPDeviceRegistrationRequest, context);
                c.e().a(context);
                String i9 = jp.co.canon.oip.android.opal.mobileatp.f.b.g().i();
                if (g.a(i9)) {
                    jp.co.canon.oip.android.opal.mobileatp.f.b.g().a();
                    i9 = jp.co.canon.oip.android.opal.mobileatp.f.b.g().i();
                    if (g.a(i9)) {
                        throw new ATPException(105, "serialNumber is empty.");
                    }
                }
                String str = i9;
                String digestName = jp.co.canon.oip.android.opal.mobileatp.f.b.g().e().getDigestName();
                String digestKey = jp.co.canon.oip.android.opal.mobileatp.f.b.g().e().getDigestKey();
                String registrationServerName = jp.co.canon.oip.android.opal.mobileatp.f.b.g().e().getRegistrationServerName();
                b.a(3, "Regist Url = " + registrationServerName);
                if (digestName != null) {
                    if (!digestName.isEmpty()) {
                        if (digestKey != null) {
                            if (digestKey.isEmpty()) {
                            }
                            jp.co.canon.oip.android.opal.mobileatp.f.b.g().a((jp.co.canon.oip.android.opal.mobileatp.a.b.c) new jp.co.canon.oip.android.opal.mobileatp.a.a.b(str, aTPDeviceRegistrationRequest, digestName, digestKey, aTPProxySetting).j());
                            b.a(3, "OK");
                            ATPResult aTPResult = new ATPResult(0, 0, "");
                            c.b();
                            jp.co.canon.oip.android.opal.mobileatp.f.b.c();
                            return aTPResult;
                        }
                    }
                }
                String[] split = registrationServerName.split(Pattern.quote(CNMLJCmnUtil.DOT));
                if (split != null && split[0].matches("^https?://.*-uw2.*")) {
                    digestName = jp.co.canon.oip.android.opal.mobileatp.util.c.d("v0L/aHx1lKH3q1CiYjZn7fYmq5K9f5SWo5btM1pN6kywriLDkS/QN+zd6TLNBdAu4RSmla50ivbhOn6d3CI9IbXHHoE=");
                    digestKey = jp.co.canon.oip.android.opal.mobileatp.util.c.d("2kBIblhVWVzlX2J7DNRDuLY4/PVjYl++jt8XuWPTOOeDMeo6");
                } else if (split == null || !split[0].matches("^https?://.*-cn1w.*")) {
                    digestName = jp.co.canon.oip.android.opal.mobileatp.util.c.d("P7Xme5JHlB7mIV06MU4gXVzopss1T6T4PL0VyRSuVhuL1uGSuuIvZIbVi+kpfqeM2D/ATtutDvkfotr+ai3eobYFaDc=");
                    digestKey = jp.co.canon.oip.android.opal.mobileatp.util.c.d("6ag3Ko1Z+PfLTJsk95clZ9vgUMxDK183f8rP63v1SY7copKr");
                } else {
                    digestName = jp.co.canon.oip.android.opal.mobileatp.util.c.d("cZAct+dlLYxI5yxNz/qgvuRklXWbTyi/3bYmVgOWfH2j40ckAAmxtwVKoJab0KVkbAOjnGgA1pmqJN0QQJcqjWXB2PY=");
                    digestKey = jp.co.canon.oip.android.opal.mobileatp.util.c.d("aZCTbvEAXgEueIftwt4KqqotpJPRzQdRrPBg/s67RCueU0DL");
                }
                jp.co.canon.oip.android.opal.mobileatp.f.b.g().a((jp.co.canon.oip.android.opal.mobileatp.a.b.c) new jp.co.canon.oip.android.opal.mobileatp.a.a.b(str, aTPDeviceRegistrationRequest, digestName, digestKey, aTPProxySetting).j());
                b.a(3, "OK");
                ATPResult aTPResult2 = new ATPResult(0, 0, "");
                c.b();
                jp.co.canon.oip.android.opal.mobileatp.f.b.c();
                return aTPResult2;
            } catch (ATPException e10) {
                b.a(3, "NG");
                ATPResult aTPResult3 = new ATPResult(e10);
                c.b();
                jp.co.canon.oip.android.opal.mobileatp.f.b.c();
                return aTPResult3;
            }
        } catch (Throwable th) {
            c.b();
            jp.co.canon.oip.android.opal.mobileatp.f.b.c();
            throw th;
        }
    }

    private void a(ATPDeviceRegistrationRequest aTPDeviceRegistrationRequest, Context context, ATPProxySetting aTPProxySetting) {
        if (aTPDeviceRegistrationRequest == null) {
            b.a(3, "1: null");
        } else {
            b.a(3, "1: not null");
            if (g.a(aTPDeviceRegistrationRequest.getRealm())) {
                b.a(3, "2: null");
            } else {
                b.a(3, "2: " + aTPDeviceRegistrationRequest.getRealm());
            }
            if (g.a(aTPDeviceRegistrationRequest.getClientName())) {
                b.a(3, "3: null");
            } else {
                b.a(3, "3: " + aTPDeviceRegistrationRequest.getClientName());
            }
            if (g.a(aTPDeviceRegistrationRequest.getClientDescription())) {
                b.a(3, "4: null");
            } else {
                b.a(3, "4: " + aTPDeviceRegistrationRequest.getClientDescription());
            }
        }
        if (context == null) {
            b.a(3, "5: null");
        } else {
            b.a(3, "5: not null");
        }
        if (aTPProxySetting == null) {
            b.a(3, "8: null");
            return;
        }
        b.a(3, "8: " + aTPProxySetting.getHost());
        b.a(3, "9: " + aTPProxySetting.getPort());
        b.a(3, "10: " + aTPProxySetting.getUser());
        b.a(3, "11: " + aTPProxySetting.getPassword());
    }
}
