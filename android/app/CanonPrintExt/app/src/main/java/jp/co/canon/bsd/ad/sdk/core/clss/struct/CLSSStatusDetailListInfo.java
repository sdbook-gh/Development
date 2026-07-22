package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSStatusDetailListInfo {
    public String errorCode;
    public int itemId;
    public int severity;
    public int summary;

    public CLSSStatusDetailListInfo() {
        init();
    }

    public void init() {
        set(Integer.MAX_VALUE, 65535, 65535, null);
    }

    public void set(int i9, int i10, int i11, String str) {
        this.itemId = i9;
        this.severity = i10;
        this.summary = i11;
        this.errorCode = str;
    }
}
