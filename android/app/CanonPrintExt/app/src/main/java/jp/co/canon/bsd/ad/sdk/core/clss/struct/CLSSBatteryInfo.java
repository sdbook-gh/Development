package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSBatteryInfo {
    public int level;
    public int[] status;

    public CLSSBatteryInfo() {
        init();
    }

    public void init() {
        set(null, -1);
    }

    public void set(int[] iArr, int i9) {
        this.status = iArr;
        this.level = i9;
    }
}
