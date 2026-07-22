package jp.co.canon.oip.android.opal.mobileatp.f;

import java.util.Properties;
import java.util.UUID;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;

/* compiled from: ATPMobileATPInfo.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class a {

    /* renamed from: b, reason: collision with root package name */
    public static final String f16919b = "serial_number";

    /* renamed from: a, reason: collision with root package name */
    private String f16920a = "";

    private void a(String str) {
        this.f16920a = str;
    }

    public void b() {
        a();
        String replaceAll = UUID.randomUUID().toString().replaceAll(jp.co.canon.oip.android.opal.mobileatp.util.b.f16930b, "");
        this.f16920a = replaceAll;
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, replaceAll);
    }

    public Properties c() {
        Properties properties = new Properties();
        properties.setProperty(f16919b, this.f16920a);
        return properties;
    }

    public String d() {
        return this.f16920a;
    }

    public void a(Properties properties) {
        a();
        if (properties == null || properties.size() == 0) {
            throw new ATPException(106, "mobileATP property is empty.");
        }
        if (!properties.containsKey(f16919b)) {
            throw new ATPException(106, "serialNumber is empty.");
        }
        this.f16920a = properties.getProperty(f16919b);
    }

    private void a() {
        this.f16920a = "";
    }
}
