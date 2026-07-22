package jp.co.canon.oip.android.opal.mobileatp.c;

import java.io.File;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import jp.co.canon.oip.android.opal.mobileatp.util.g;

/* compiled from: ATPAbstractFile.java */
/* loaded from: /mnt/f/print/classes3.dex */
public abstract class a {

    /* renamed from: a, reason: collision with root package name */
    private File f16883a;

    /* renamed from: b, reason: collision with root package name */
    private String f16884b;

    private a() {
        this.f16883a = null;
        this.f16884b = null;
    }

    private void a(String str, String str2) {
        if (g.a(str) || g.a(str2)) {
            return;
        }
        this.f16883a = jp.co.canon.oip.android.opal.mobileatp.util.d.a(str);
        this.f16884b = str2;
    }

    public abstract Object a(File file);

    public abstract void a(File file, Object obj);

    public File b() {
        return this.f16883a;
    }

    public String c() {
        return this.f16884b;
    }

    public Object b(String str) {
        if (g.a(str)) {
            throw new ATPException(1001);
        }
        return a(jp.co.canon.oip.android.opal.mobileatp.util.d.b(this.f16883a, str));
    }

    public a(String str, String str2) {
        this();
        a(str, str2);
    }

    public void a(String str, Object obj) {
        if (!g.a(str)) {
            a(jp.co.canon.oip.android.opal.mobileatp.util.d.a(this.f16883a, str), obj);
            return;
        }
        throw new ATPException(1001);
    }

    public void a() {
        a(this.f16884b);
    }

    public void a(String str) {
        if (this.f16883a != null) {
            if (!g.a(str)) {
                jp.co.canon.oip.android.opal.mobileatp.util.d.a(jp.co.canon.oip.android.opal.mobileatp.util.d.b(this.f16883a, str));
                return;
            }
            throw new ATPException(1001);
        }
        throw new ATPException(1000);
    }
}
