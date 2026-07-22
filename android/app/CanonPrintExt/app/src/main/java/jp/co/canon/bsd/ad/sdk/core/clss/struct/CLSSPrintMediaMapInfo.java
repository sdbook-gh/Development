package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSPrintMediaMapInfo {
    public int id_commandSort_back;
    public int id_commandSort_front;
    public int papertypeID_back;
    public int papertypeID_front;

    public CLSSPrintMediaMapInfo() {
        init();
    }

    public void init() {
        set(65535, 65535, 65535, 65535);
    }

    public void set(int i9, int i10, int i11, int i12) {
        this.papertypeID_front = i9;
        this.papertypeID_back = i10;
        this.id_commandSort_front = i11;
        this.id_commandSort_back = i12;
    }
}
