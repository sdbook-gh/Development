package jp.co.canon.oip.android.opal.mobileatp.b;

/* compiled from: ATPConfig.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class a {

    /* renamed from: a, reason: collision with root package name */
    private static a f16880a = null;

    /* renamed from: b, reason: collision with root package name */
    public static final boolean f16881b = true;

    /* renamed from: c, reason: collision with root package name */
    public static final int f16882c = -1;

    private a() {
    }

    public static synchronized a a() {
        a aVar;
        synchronized (a.class) {
            try {
                if (f16880a == null) {
                    f16880a = new a();
                }
                aVar = f16880a;
            } catch (Throwable th) {
                throw th;
            }
        }
        return aVar;
    }
}
