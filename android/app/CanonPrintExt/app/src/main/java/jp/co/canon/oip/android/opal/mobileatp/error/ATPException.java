package jp.co.canon.oip.android.opal.mobileatp.error;

import jp.co.canon.oip.android.opal.mobileatp.d.b;

/* loaded from: /mnt/f/print/classes3.dex */
/** Checked in original APK; RuntimeException here so cleaned jadx sources compile. */
public class ATPException extends RuntimeException {

    /* renamed from: e, reason: collision with root package name */
    private static final long f16914e = 1;

    /* renamed from: a, reason: collision with root package name */
    private int f16915a;

    /* renamed from: b, reason: collision with root package name */
    private int f16916b;

    /* renamed from: c, reason: collision with root package name */
    private String f16917c;

    /* renamed from: d, reason: collision with root package name */
    private String f16918d;

    private ATPException() {
        this.f16915a = 0;
        this.f16916b = 0;
        this.f16917c = "";
        this.f16918d = "";
    }

    public String getErrorCode() {
        return this.f16917c;
    }

    public String getErrorDescription() {
        return this.f16918d;
    }

    public int getHttpStatusCode() {
        return this.f16916b;
    }

    public int getStatus() {
        return this.f16915a;
    }

    public ATPException(int i9) {
        this();
        this.f16915a = i9;
        b.a(i9, this);
    }

    public ATPException(int i9, String str) {
        super(str);
        this.f16916b = 0;
        this.f16917c = "";
        this.f16918d = "";
        this.f16915a = i9;
        b.a(i9, this);
    }

    public ATPException(int i9, Throwable th) {
        super(th);
        this.f16916b = 0;
        this.f16917c = "";
        this.f16918d = "";
        this.f16915a = i9;
        b.a(i9, this);
    }

    public ATPException(int i9, String str, Throwable th) {
        super(str, th);
        this.f16916b = 0;
        this.f16917c = "";
        this.f16918d = "";
        this.f16915a = i9;
        b.a(i9, this, th);
    }

    public ATPException(int i9, int i10, String str) {
        this();
        this.f16915a = i9;
        this.f16916b = i10;
        if (str != null) {
            this.f16917c = str;
        }
        b.a(i9, this);
    }

    public ATPException(int i9, int i10, String str, String str2) {
        this();
        this.f16915a = i9;
        this.f16916b = i10;
        this.f16918d = str2;
        if (str != null) {
            this.f16917c = str;
        }
        b.a(i9, this);
    }
}
