package jp.co.canon.oip.android.opal.mobileatp.f;

import java.util.Properties;
import jp.co.canon.oip.android.opal.mobileatp.ATPCAMSConnectSetting;
import jp.co.canon.oip.android.opal.mobileatp.a.b.c;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import jp.co.canon.oip.android.opal.mobileatp.util.g;

/* compiled from: ATPMobileATPSystem.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class b {

    /* renamed from: d, reason: collision with root package name */
    private static b f16921d;

    /* renamed from: a, reason: collision with root package name */
    private a f16922a = null;

    /* renamed from: b, reason: collision with root package name */
    private c f16923b = null;

    /* renamed from: c, reason: collision with root package name */
    private ATPCAMSConnectSetting f16924c;

    private b() {
        a((ATPCAMSConnectSetting) null);
    }

    public static synchronized void c() {
        synchronized (b.class) {
            f16921d = null;
        }
    }

    public static synchronized b g() {
        b bVar;
        synchronized (b.class) {
            try {
                if (f16921d == null) {
                    f16921d = new b();
                }
                bVar = f16921d;
            } catch (Throwable th) {
                throw th;
            }
        }
        return bVar;
    }

    public void a(c cVar) {
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "start");
        if (cVar == null) {
            throw new ATPException(102, "deviceCredential is null.");
        }
        b();
        jp.co.canon.oip.android.opal.mobileatp.c.c.e().a(cVar.e());
    }

    public void b() {
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "start");
        this.f16923b = null;
        jp.co.canon.oip.android.opal.mobileatp.c.c.e().a();
    }

    public void d() {
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "start");
        this.f16922a = null;
        jp.co.canon.oip.android.opal.mobileatp.c.c.e().c();
    }

    public ATPCAMSConnectSetting e() {
        return this.f16924c;
    }

    public c f() {
        if (this.f16923b == null) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "from file.");
            try {
                Properties f10 = jp.co.canon.oip.android.opal.mobileatp.c.c.e().f();
                if (f10 != null && f10.size() > 0) {
                    c cVar = new c();
                    this.f16923b = cVar;
                    cVar.a(f10);
                }
            } catch (ATPException e10) {
                this.f16923b = null;
                throw e10;
            } catch (Exception unused) {
                this.f16923b = null;
                throw new ATPException(102, "getDeviceCredential is invalid.");
            }
        } else {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "from cache.");
        }
        return this.f16923b;
    }

    public a h() {
        if (this.f16922a == null) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "from file.");
            try {
                Properties g10 = jp.co.canon.oip.android.opal.mobileatp.c.c.e().g();
                if (g10 != null && g10.size() > 0) {
                    a aVar = new a();
                    this.f16922a = aVar;
                    aVar.a(g10);
                }
            } catch (ATPException e10) {
                throw e10;
            } catch (Exception unused) {
                throw new ATPException(102, "getMobileATPInfo is invalid.");
            }
        } else {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "from cache.");
        }
        return this.f16922a;
    }

    public String i() {
        a h9 = null;
        try {
            h9 = g().h();
        } catch (ATPException unused) {
        }
        String str = (h9 != null) ? h9.d() : "";
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, str);
        return str;
    }

    public void j() {
        String i9 = i();
        if (!g.a(i9)) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, i9);
        } else {
            try {
                g().d();
                g().b();
            } catch (ATPException unused) {
            }
            throw new ATPException(101, "serialNumber check error.");
        }
    }

    public void a() {
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "start");
        d();
        a aVar = new a();
        aVar.b();
        jp.co.canon.oip.android.opal.mobileatp.c.c.e().b(aVar.c());
    }

    public void a(ATPCAMSConnectSetting aTPCAMSConnectSetting) {
        if (aTPCAMSConnectSetting == null) {
            this.f16924c = new ATPCAMSConnectSetting();
        } else {
            this.f16924c = aTPCAMSConnectSetting.copy();
        }
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, this.f16924c.toString());
    }
}
