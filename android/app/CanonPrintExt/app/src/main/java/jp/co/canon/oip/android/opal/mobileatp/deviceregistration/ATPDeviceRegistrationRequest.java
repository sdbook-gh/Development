package jp.co.canon.oip.android.opal.mobileatp.deviceregistration;

/* loaded from: /mnt/f/print/classes3.dex */
public class ATPDeviceRegistrationRequest {

    /* renamed from: a, reason: collision with root package name */
    private String f16908a = "";

    /* renamed from: b, reason: collision with root package name */
    private String f16909b = "";

    /* renamed from: c, reason: collision with root package name */
    private String[] f16910c = new String[0];

    /* renamed from: d, reason: collision with root package name */
    private String[] f16911d = new String[0];

    /* renamed from: e, reason: collision with root package name */
    private String f16912e = "";

    /* renamed from: f, reason: collision with root package name */
    private String f16913f = "";

    public String getApplicationId() {
        return this.f16913f;
    }

    public String getClientDescription() {
        return this.f16909b;
    }

    public String getClientName() {
        return this.f16908a;
    }

    public String[] getDefaultScopes() {
        return this.f16911d;
    }

    public String getRealm() {
        return this.f16912e;
    }

    public String[] getScopes() {
        return this.f16910c;
    }

    public void setApplicationId(String str) {
        this.f16913f = str;
    }

    public void setClientDescription(String str) {
        this.f16909b = str;
    }

    public void setClientName(String str) {
        this.f16908a = str;
    }

    public void setDefaultScopes(String[] strArr) {
        this.f16911d = strArr;
    }

    public void setRealm(String str) {
        this.f16912e = str;
    }

    public void setScopes(String[] strArr) {
        this.f16910c = strArr;
    }
}
