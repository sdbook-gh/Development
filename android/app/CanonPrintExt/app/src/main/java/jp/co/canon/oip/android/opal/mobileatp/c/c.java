package jp.co.canon.oip.android.opal.mobileatp.c;

import android.content.Context;
import java.util.Properties;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;

/* compiled from: ATPFileManager.java */
/* loaded from: /mnt/f/print/classes3.dex */
public final class c {

    /* renamed from: c, reason: collision with root package name */
    private static c f16889c;

    /* renamed from: a, reason: collision with root package name */
    private d f16890a = null;

    /* renamed from: b, reason: collision with root package name */
    private d f16891b = null;

    private c() {
    }

    public static synchronized void b() {
        synchronized (c.class) {
            f16889c = null;
        }
    }

    public static synchronized c e() {
        c cVar;
        synchronized (c.class) {
            try {
                if (f16889c == null) {
                    f16889c = new c();
                    jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "start");
                }
                cVar = f16889c;
            } catch (Throwable th) {
                throw th;
            }
        }
        return cVar;
    }

    public void a(Context context) {
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "start");
        this.f16890a = new d(context.getFilesDir().getPath() + "/mobileATP/credentials", "deviceCredential.properties");
        this.f16891b = new d(context.getFilesDir().getPath() + "/mobileATP/", "atpinfo.properties");
    }

    public void c() {
        d dVar = this.f16891b;
        if (dVar != null) {
            dVar.a();
        }
    }

    public void d() {
        f16889c = null;
    }

    public Properties f() {
        d dVar = this.f16890a;
        if (dVar != null) {
            return dVar.d();
        }
        throw new ATPException(102, "deviceCredentialProperties is null.");
    }

    public Properties g() {
        d dVar = this.f16891b;
        if (dVar != null) {
            return dVar.d();
        }
        throw new ATPException(102, "clientInfoProperties is null.");
    }

    public void b(Properties properties) {
        if (properties == null || properties.size() <= 0) {
            throw new ATPException(1005, "mobileATPinfo is empty");
        }
        this.f16891b.a(properties);
    }

    public void a(Properties properties) {
        if (properties != null && properties.size() > 0) {
            d dVar = this.f16890a;
            if (dVar != null) {
                dVar.a(properties);
                return;
            }
            throw new ATPException(102, "deviceCredentialProperties is null.");
        }
        throw new ATPException(1005, "device credential is empty.");
    }

    public void a() {
        d dVar = this.f16890a;
        if (dVar != null) {
            dVar.a();
        }
    }
}
