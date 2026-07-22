package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import d7.InterfaceC1549a;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSPrintcolormodeInfo {
    private static final String PREF_PCMI_PRINTCOLORMODE = "_pcmi_printcolormode";
    private static final String PREF_PCMI_PRINTCOLORMODE_INTENT = "_pcmi_printcolormode_intent";

    @InterfaceC1549a(defInt = 65535, key = PREF_PCMI_PRINTCOLORMODE)
    public int printcolormode;

    @InterfaceC1549a(key = PREF_PCMI_PRINTCOLORMODE_INTENT)
    public int[] printcolormode_intent;
    public int version;

    public CLSSPrintcolormodeInfo() {
        init();
    }

    private void init() {
        set(65535, 65535, null);
    }

    public int getPrintcolormode() {
        return this.printcolormode;
    }

    public int[] getPrintcolormode_intent() {
        return this.printcolormode_intent;
    }

    public void set(int i9, int i10, int[] iArr) {
        this.version = i9;
        this.printcolormode = i10;
        this.printcolormode_intent = iArr;
    }

    public void setPrintcolormode(int i9) {
        this.printcolormode = i9;
    }

    public void setPrintcolormode_intent(int[] iArr) {
        this.printcolormode_intent = iArr;
    }
}
