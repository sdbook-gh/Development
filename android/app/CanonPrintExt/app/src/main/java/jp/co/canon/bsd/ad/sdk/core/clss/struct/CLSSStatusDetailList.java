package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSStatusDetailList {
    public int listNum;
    public CLSSStatusDetailListInfo[] statusDatailListInfo;

    public CLSSStatusDetailList() {
        init();
    }

    public void init() {
        this.statusDatailListInfo = null;
        this.listNum = 65535;
    }

    public void set(CLSSStatusDetailListInfo[] cLSSStatusDetailListInfoArr, int i9) {
        if (cLSSStatusDetailListInfoArr == null) {
            this.statusDatailListInfo = null;
        } else {
            this.statusDatailListInfo = new CLSSStatusDetailListInfo[cLSSStatusDetailListInfoArr.length];
            for (int i10 = 0; i10 < cLSSStatusDetailListInfoArr.length; i10++) {
                this.statusDatailListInfo[i10] = cLSSStatusDetailListInfoArr[i10];
            }
        }
        this.listNum = i9;
    }
}
