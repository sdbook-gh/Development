package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import d7.InterfaceC1549a;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSBorderInfo {
    private static final String PREF_BI_EXTENSION = "_bi_extension";
    private static final String PREF_BI_MARGIN = "_bi_margin";

    @InterfaceC1549a(defInt = 65535, key = PREF_BI_EXTENSION)
    public int extension;

    @InterfaceC1549a(key = PREF_BI_MARGIN)
    public int[] margin;

    public CLSSBorderInfo() {
        init();
    }

    public void init() {
        this.extension = 65535;
        this.margin = new int[4];
        int i9 = 0;
        while (true) {
            int[] iArr = this.margin;
            if (i9 >= iArr.length) {
                return;
            }
            iArr[i9] = 0;
            i9++;
        }
    }
}
