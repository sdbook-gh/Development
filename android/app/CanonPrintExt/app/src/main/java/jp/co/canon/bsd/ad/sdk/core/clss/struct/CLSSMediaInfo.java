package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import d7.InterfaceC1549a;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSMediaInfo {
    private static final String PREF_MI_BORDERLESS_PRINT_AVAILABLE = "_mi_borderless_print_available";
    private static final String PREF_MI_BORDER_PRINT_AVAILABLE = "_mi_border_print_available";
    private static final String PREF_MI_COLOR_MODE_ID = "_mi_color_mode_id";
    private static final String PREF_MI_DUPLEX_ID = "_mi_duplex_id";
    private static final String PREF_MI_PAPER_TYPE_ID = "_mi_paper_type_id";

    @InterfaceC1549a(key = PREF_MI_BORDERLESS_PRINT_AVAILABLE)
    public boolean borderlessprintAvailable;

    @InterfaceC1549a(key = PREF_MI_BORDER_PRINT_AVAILABLE)
    public boolean borderprintAvailable;

    @InterfaceC1549a(key = PREF_MI_COLOR_MODE_ID)
    public int[] colormodeID;

    @InterfaceC1549a(key = PREF_MI_DUPLEX_ID)
    public int[] duplexID;

    @InterfaceC1549a(defInt = 65535, key = PREF_MI_PAPER_TYPE_ID)
    public int papertypeID;

    public CLSSMediaInfo() {
        init();
    }

    private void init() {
        set(65535, false, false, null, null);
    }

    public void set(int i9, boolean z9, boolean z10, int[] iArr, int[] iArr2) {
        this.papertypeID = i9;
        this.borderprintAvailable = z9;
        this.borderlessprintAvailable = z10;
        if (iArr == null) {
            this.colormodeID = null;
        } else {
            this.colormodeID = new int[iArr.length];
            for (int i10 = 0; i10 < iArr.length; i10++) {
                this.colormodeID[i10] = iArr[i10];
            }
        }
        if (iArr2 == null) {
            this.duplexID = null;
        } else {
            this.duplexID = new int[iArr2.length];
            for (int i11 = 0; i11 < iArr2.length; i11++) {
                this.duplexID[i11] = iArr2[i11];
            }
        }
    }
}
