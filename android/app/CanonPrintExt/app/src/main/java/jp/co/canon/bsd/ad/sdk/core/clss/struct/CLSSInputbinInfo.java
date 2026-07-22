package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import d7.InterfaceC1549a;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSInputbinInfo {
    private static final int MARGIN_MAX = 4;
    private static final String PREF_IBI_ID_COMMANDSORT = "_ibi_id_commandsort";
    private static final String PREF_IBI_INPUTBIN = "_ibi_inputbin";
    private static final String PREF_IBI_INPUTBINID = "_ibi_inputbinid";
    private static final String PREF_IBI_MARGIN_BORDER = "_ibi_margin_border";
    private static final String PREF_IBI_MARGIN_BORDERLESS = "_ibi_imargin_borderless";
    private static final String PREF_IBI_PAPERSIZE_CUSTOM_HEIGHT = "_ibi_papersize_custom_height";
    private static final String PREF_IBI_PAPERSIZE_CUSTOM_HEIGHT_RECOMMENDED = "_ibi_papersize_custom_height_recommended";
    private static final String PREF_IBI_PAPERSIZE_CUSTOM_WIDTH = "_ibi_papersize_custom_width";
    private static final int RANGE_MAX = 2;

    @InterfaceC1549a(defInt = 65535, key = PREF_IBI_ID_COMMANDSORT)
    public int id_commandSort;

    @InterfaceC1549a(defInt = 65535, key = PREF_IBI_INPUTBIN)
    public int inputbin;

    @InterfaceC1549a(defInt = 65535, key = PREF_IBI_INPUTBINID)
    public int inputbinid;

    @InterfaceC1549a(key = PREF_IBI_MARGIN_BORDER)
    public long[] margin_border;

    @InterfaceC1549a(key = PREF_IBI_MARGIN_BORDERLESS)
    public int[] margin_borderless;

    @InterfaceC1549a(key = PREF_IBI_PAPERSIZE_CUSTOM_HEIGHT)
    public long[] papersize_custom_height;

    @InterfaceC1549a(key = PREF_IBI_PAPERSIZE_CUSTOM_HEIGHT_RECOMMENDED)
    public long[] papersize_custom_height_recommended;

    @InterfaceC1549a(key = PREF_IBI_PAPERSIZE_CUSTOM_WIDTH)
    public long[] papersize_custom_width;

    public CLSSInputbinInfo() {
        init();
    }

    private void init() {
        long[] jArr = new long[4];
        int[] iArr = new int[4];
        long[] jArr2 = new long[2];
        for (int i9 = 0; i9 < 4; i9++) {
            iArr[i9] = Integer.MAX_VALUE;
            jArr[i9] = 4294967295L;
        }
        for (int i10 = 0; i10 < 2; i10++) {
            jArr2[i10] = 4294967295L;
        }
        set(65535, 65535, 65535, jArr2, jArr, iArr, jArr2, jArr2);
    }

    public int getId_commandSort() {
        return this.id_commandSort;
    }

    public int getInputbin() {
        return this.inputbin;
    }

    public int getInputbinid() {
        return this.inputbinid;
    }

    public long[] getMargin_border() {
        return this.margin_border;
    }

    public int[] getMargin_borderless() {
        return this.margin_borderless;
    }

    public long[] getPapersize_custom_height() {
        return this.papersize_custom_height;
    }

    public long[] getPapersize_custom_height_recommended() {
        return this.papersize_custom_height_recommended;
    }

    public long[] getPapersize_custom_width() {
        return this.papersize_custom_width;
    }

    public void set(int i9, int i10, int i11, long[] jArr, long[] jArr2, int[] iArr, long[] jArr3, long[] jArr4) {
        this.id_commandSort = i9;
        this.inputbin = i10;
        this.inputbinid = i11;
        this.papersize_custom_height_recommended = jArr;
        this.margin_border = jArr2;
        this.margin_borderless = iArr;
        this.papersize_custom_width = jArr3;
        this.papersize_custom_height = jArr4;
    }

    public void setId_commandSort(int i9) {
        this.id_commandSort = i9;
    }

    public void setInputbin(int i9) {
        this.inputbin = i9;
    }

    public void setInputbinid(int i9) {
        this.inputbinid = i9;
    }

    public void setMargin_border(long[] jArr) {
        this.margin_border = jArr;
    }

    public void setMargin_borderless(int[] iArr) {
        this.margin_borderless = iArr;
    }

    public void setPapersize_custom_height(long[] jArr) {
        this.papersize_custom_height = jArr;
    }

    public void setPapersize_custom_height_recommended(long[] jArr) {
        this.papersize_custom_height_recommended = jArr;
    }

    public void setPapersize_custom_width(long[] jArr) {
        this.papersize_custom_width = jArr;
    }
}
