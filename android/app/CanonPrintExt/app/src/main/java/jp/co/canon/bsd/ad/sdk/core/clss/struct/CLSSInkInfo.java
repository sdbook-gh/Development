package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSInkInfo {
    public String color;
    public int inkstatus;
    public int level;
    public String model;
    public int order;

    public CLSSInkInfo() {
        init();
    }

    public void init() {
        set(null, null, 65535, -1, -1);
    }

    public boolean isInkInfoEmpty() {
        return this.order == -1;
    }

    public void set(String str, String str2, int i9, int i10, int i11) {
        this.model = str;
        this.color = str2;
        this.inkstatus = i9;
        this.level = i10;
        this.order = i11;
    }
}
