package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSDeviceInkInfo {
    private String mColor;
    private int mOrder;
    private String mStatus;

    public CLSSDeviceInkInfo() {
        set(null, null, -1);
    }

    public String getColor() {
        return this.mColor;
    }

    public int getOrder() {
        return this.mOrder;
    }

    public String getStatus() {
        return this.mStatus;
    }

    public void set(String str, String str2, int i9) {
        this.mColor = str;
        this.mStatus = str2;
        this.mOrder = i9;
    }

    public CLSSDeviceInkInfo(String str, String str2, int i9) {
        this.mColor = str;
        this.mStatus = str2;
        this.mOrder = i9;
    }
}
