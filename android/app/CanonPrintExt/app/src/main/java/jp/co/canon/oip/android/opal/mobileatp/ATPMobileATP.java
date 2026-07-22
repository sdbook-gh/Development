package jp.co.canon.oip.android.opal.mobileatp;

import android.content.Context;
import jp.co.canon.oip.android.opal.mobileatp.c.c;
import jp.co.canon.oip.android.opal.mobileatp.d.b;
import jp.co.canon.oip.android.opal.mobileatp.deviceregistration.ATPDeviceRegistrationRequest;
import jp.co.canon.oip.android.opal.mobileatp.token.ATPResultAccessToken;
import jp.co.canon.oip.android.opal.mobileatp.token.a;

/* loaded from: /mnt/f/print/classes3.dex */
public class ATPMobileATP {
    private static final Object MATP_LOCK = new Object();

    public ATPMobileATP() {
    }

    public ATPResultAccessToken getAccessToken(String[] strArr, Context context, String str, ATPProxySetting aTPProxySetting) {
        ATPResultAccessToken a10;
        synchronized (MATP_LOCK) {
            b.a(3, "start");
            a10 = new a().a(strArr, context, str, aTPProxySetting);
            b.a(3, a10.getResultCode());
            b.a(3, a10.getHttpStatusCode());
            b.a(3, a10.getErrorCode());
            b.a(3, a10.getAccessToken());
            b.a(3, a10.getTokenType());
            b.a(3, a10.getExpiresIn());
            b.a(3, a10.getScope());
        }
        return a10;
    }

    public String getSerialNumber(Context context) {
        String str = null;
        try {
            c.e().a(context);
            String i9 = jp.co.canon.oip.android.opal.mobileatp.f.b.g().i();
            if (i9 != null && !i9.isEmpty()) {
                str = i9;
            }
        } catch (Exception e10) {
            b.a(3, e10.getMessage());
        } finally {
            c.b();
            jp.co.canon.oip.android.opal.mobileatp.f.b.c();
        }
        return str;
    }

    public ATPResult registerDevice(ATPDeviceRegistrationRequest aTPDeviceRegistrationRequest, Context context, ATPProxySetting aTPProxySetting) {
        ATPResult b10;
        synchronized (MATP_LOCK) {
            b.a(3, "start");
            b10 = new jp.co.canon.oip.android.opal.mobileatp.deviceregistration.a().b(aTPDeviceRegistrationRequest, context, aTPProxySetting);
            b.a(3, b10.getResultCode());
            b.a(3, b10.getHttpStatusCode());
            b.a(3, b10.getErrorCode());
        }
        return b10;
    }

    public void setCAMSConnectSetting(ATPCAMSConnectSetting aTPCAMSConnectSetting) {
        jp.co.canon.oip.android.opal.mobileatp.f.b.g().a(aTPCAMSConnectSetting);
    }

    public ATPMobileATP(ATPCAMSConnectSetting aTPCAMSConnectSetting) {
        this();
        setCAMSConnectSetting(aTPCAMSConnectSetting);
    }
}
