package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSDeviceProcessingStatus {
    private int mProcessingRemain;
    private int mProcessingTotal;
    private int mProcessingUnit;

    public CLSSDeviceProcessingStatus() {
        set(-1, -1, 65535);
    }

    public int getProcessingRemain() {
        return this.mProcessingRemain;
    }

    public int getProcessingTotal() {
        return this.mProcessingTotal;
    }

    public int getProcessingUnit() {
        return this.mProcessingUnit;
    }

    public void set(int i9, int i10, int i11) {
        this.mProcessingTotal = i9;
        this.mProcessingRemain = i10;
        this.mProcessingUnit = i11;
    }

    public CLSSDeviceProcessingStatus(int i9, int i10, int i11) {
        this.mProcessingTotal = i9;
        this.mProcessingRemain = i10;
        this.mProcessingUnit = i11;
    }
}
