package jp.co.canon.oip.android.opal.mobileatp.token;

import jp.co.canon.oip.android.opal.mobileatp.ATPResult;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;

/* loaded from: /mnt/f/print/classes3.dex */
public class ATPResultAccessToken extends ATPResult {

    /* renamed from: e, reason: collision with root package name */
    private String f16925e;

    /* renamed from: f, reason: collision with root package name */
    private String f16926f;

    /* renamed from: g, reason: collision with root package name */
    private int f16927g;

    /* renamed from: h, reason: collision with root package name */
    private String f16928h;

    public ATPResultAccessToken(int i9, int i10, String str, String str2, String str3, int i11, String str4) {
        super(i9, i10, str);
        this.f16925e = str2;
        this.f16926f = str3;
        this.f16927g = i11;
        this.f16928h = str4;
    }

    public String getAccessToken() {
        return this.f16925e;
    }

    public int getExpiresIn() {
        return this.f16927g;
    }

    public String getScope() {
        return this.f16928h;
    }

    public String getTokenType() {
        return this.f16926f;
    }

    public ATPResultAccessToken(ATPException aTPException) {
        super(aTPException);
        this.f16925e = "";
        this.f16926f = "";
        this.f16927g = 0;
        this.f16928h = "";
    }
}
