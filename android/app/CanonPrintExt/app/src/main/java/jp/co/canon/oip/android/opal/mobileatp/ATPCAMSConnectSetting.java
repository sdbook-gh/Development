package jp.co.canon.oip.android.opal.mobileatp;

import A6.r;
import jp.co.canon.oip.android.opal.mobileatp.a.a.e;

/* loaded from: /mnt/f/print/classes3.dex */
public class ATPCAMSConnectSetting {

    /* renamed from: a, reason: collision with root package name */
    private int f16810a;

    /* renamed from: b, reason: collision with root package name */
    private long f16811b;

    /* renamed from: c, reason: collision with root package name */
    private int f16812c;

    /* renamed from: d, reason: collision with root package name */
    private int f16813d;

    /* renamed from: e, reason: collision with root package name */
    private String f16814e;

    /* renamed from: f, reason: collision with root package name */
    private String f16815f;

    /* renamed from: g, reason: collision with root package name */
    private String f16816g;

    /* renamed from: h, reason: collision with root package name */
    private String f16817h;

    /* renamed from: i, reason: collision with root package name */
    private String f16818i;

    public ATPCAMSConnectSetting() {
        this.f16810a = 3;
        this.f16811b = 1000L;
        this.f16812c = 5000;
        this.f16813d = 5000;
        this.f16814e = e.f16862s;
        this.f16815f = "";
        this.f16816g = "";
        this.f16817h = "";
        this.f16818i = "";
    }

    public ATPCAMSConnectSetting copy() {
        ATPCAMSConnectSetting aTPCAMSConnectSetting = new ATPCAMSConnectSetting();
        aTPCAMSConnectSetting.setRetryCount(this.f16810a);
        aTPCAMSConnectSetting.setRetryInterval(this.f16811b);
        aTPCAMSConnectSetting.setConTimeout(this.f16812c);
        aTPCAMSConnectSetting.setSoTimeout(this.f16813d);
        aTPCAMSConnectSetting.setUserAgent(this.f16814e);
        aTPCAMSConnectSetting.setRegistrationServerName(this.f16815f);
        aTPCAMSConnectSetting.setTokenServerName(this.f16816g);
        aTPCAMSConnectSetting.setDigestName(this.f16818i);
        aTPCAMSConnectSetting.setDigestKey(this.f16817h);
        return aTPCAMSConnectSetting;
    }

    public int getConTimeout() {
        return this.f16812c;
    }

    public String getDigestKey() {
        return this.f16817h;
    }

    public String getDigestName() {
        return this.f16818i;
    }

    public String getRegistrationServerName() {
        return this.f16815f;
    }

    public int getRetryCount() {
        return this.f16810a;
    }

    public long getRetryInterval() {
        return this.f16811b;
    }

    public int getSoTimeout() {
        return this.f16813d;
    }

    public String getTokenServerName() {
        return this.f16816g;
    }

    public String getUserAgent() {
        return this.f16814e;
    }

    public void setConTimeout(int i9) {
        this.f16812c = i9;
    }

    public void setDigestKey(String str) {
        this.f16817h = str;
    }

    public void setDigestName(String str) {
        this.f16818i = str;
    }

    public void setRegistrationServerName(String str) {
        this.f16815f = str;
    }

    public void setRetryCount(int i9) {
        this.f16810a = i9;
    }

    public void setRetryInterval(long j9) {
        this.f16811b = j9;
    }

    public void setSoTimeout(int i9) {
        this.f16813d = i9;
    }

    public void setTokenServerName(String str) {
        this.f16816g = str;
    }

    public void setUserAgent(String str) {
        this.f16814e = str;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder("[retryCount=");
        sb.append(this.f16810a);
        sb.append(", retryInterval=");
        sb.append(this.f16811b);
        sb.append(", conTimeout=");
        sb.append(this.f16812c);
        sb.append(", soTimeout=");
        sb.append(this.f16813d);
        sb.append(", UserAgent=");
        sb.append(this.f16814e);
        sb.append(", registrationServer=");
        sb.append(this.f16815f);
        sb.append(", tokenServer=");
        sb.append(this.f16816g);
        sb.append(", digestName=");
        return r.c(']', this.f16818i, sb);
    }

    public ATPCAMSConnectSetting(int i9, long j9, int i10, int i11) {
        this.f16814e = e.f16862s;
        this.f16815f = "";
        this.f16816g = "";
        this.f16817h = "";
        this.f16818i = "";
        this.f16810a = i9;
        this.f16811b = j9;
        this.f16812c = i10;
        this.f16813d = i11;
    }

    public ATPCAMSConnectSetting(int i9, long j9, int i10, int i11, String str) {
        this(i9, j9, i10, i11);
        this.f16814e = str;
    }

    public ATPCAMSConnectSetting(int i9, long j9, int i10, int i11, String str, String str2, String str3) {
        this(i9, j9, i10, i11, str);
        this.f16815f = str2;
        this.f16816g = str3;
    }

    public ATPCAMSConnectSetting(int i9, long j9, int i10, int i11, String str, String str2, String str3, String str4, String str5) {
        this(i9, j9, i10, i11, str, str2, str3);
        this.f16818i = str4;
        this.f16817h = str5;
    }
}
