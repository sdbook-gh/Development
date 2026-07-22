package jp.co.canon.oip.android.opal.mobileatp;

/* loaded from: /mnt/f/print/classes3.dex */
public class ATPProxySetting {

    /* renamed from: a, reason: collision with root package name */
    private String f16819a;

    /* renamed from: b, reason: collision with root package name */
    private int f16820b;

    /* renamed from: c, reason: collision with root package name */
    private String f16821c;

    /* renamed from: d, reason: collision with root package name */
    private String f16822d;

    public ATPProxySetting() {
    }

    public boolean enableAuthentication() {
        String str;
        String str2 = this.f16821c;
        return str2 != null && str2.length() > 0 && (str = this.f16822d) != null && str.length() > 0;
    }

    public String getHost() {
        return this.f16819a;
    }

    public String getPassword() {
        return this.f16822d;
    }

    public int getPort() {
        return this.f16820b;
    }

    public String getUser() {
        return this.f16821c;
    }

    public void setHost(String str) {
        this.f16819a = str;
    }

    public void setPassword(String str) {
        this.f16822d = str;
    }

    public void setPort(int i9) {
        this.f16820b = i9;
    }

    public void setUser(String str) {
        this.f16821c = str;
    }

    public ATPProxySetting(String str, int i9, String str2, String str3) {
        this.f16819a = str;
        this.f16820b = i9;
        this.f16821c = str2;
        this.f16822d = str3;
    }
}
