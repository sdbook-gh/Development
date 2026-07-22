package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import d7.InterfaceC1549a;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSPaperSizeInfo {
    private static final int MARGIN_MAX = 4;
    private static final String PREF_PSI_AVAILABLEINPUTBIN = "_psi_availableinputbin";
    private static final String PREF_PSI_BORDERLESS_PRINT_LENGTH = "_psi_borderless_print_length";
    private static final String PREF_PSI_BORDERLESS_PRINT_WIDTH = "_psi_borderless_print_width";
    private static final String PREF_PSI_BORDER_PRINT_LENGTH = "_psi_border_print_length";
    private static final String PREF_PSI_BORDER_PRINT_WIDTH = "_psi_border_print_width";
    private static final String PREF_PSI_DUPLEX_PRINT_LENGTH = "_psi_duplex_print_length";
    private static final String PREF_PSI_DUPLEX_PRINT_WIDTH = "_psi_duplex_print_width";
    private static final String PREF_PSI_MARGIN_BORDER = "_psi_margin_border";
    private static final String PREF_PSI_MARGIN_BORDERLESS = "_psi_margin_borderless";
    private static final String PREF_PSI_MARGIN_DUPLEX = "_psi_margin_duplex";
    private static final String PREF_PSI_PAPER_SIZE_ID = "_psi_paper_size_id";

    @InterfaceC1549a(key = PREF_PSI_AVAILABLEINPUTBIN)
    public int[] availableinputbin;

    @InterfaceC1549a(defInt = 65535, key = PREF_PSI_BORDERLESS_PRINT_LENGTH)
    public int borderlessprintLength;

    @InterfaceC1549a(defInt = 65535, key = PREF_PSI_BORDERLESS_PRINT_WIDTH)
    public int borderlessprintWidth;

    @InterfaceC1549a(defInt = 65535, key = PREF_PSI_BORDER_PRINT_LENGTH)
    public int borderprintLength;

    @InterfaceC1549a(defInt = 65535, key = PREF_PSI_BORDER_PRINT_WIDTH)
    public int borderprintWidth;

    @InterfaceC1549a(defInt = 65535, key = PREF_PSI_DUPLEX_PRINT_LENGTH)
    public int duplexprintLength;

    @InterfaceC1549a(defInt = 65535, key = PREF_PSI_DUPLEX_PRINT_WIDTH)
    public int duplexprintWidth;

    @InterfaceC1549a(key = PREF_PSI_MARGIN_BORDER)
    public int[] marginBorder = new int[4];

    @InterfaceC1549a(key = PREF_PSI_MARGIN_BORDERLESS)
    public int[] marginBorderless = new int[4];

    @InterfaceC1549a(key = PREF_PSI_MARGIN_DUPLEX)
    public int[] marginDuplex = new int[4];

    @InterfaceC1549a(defInt = 65535, key = PREF_PSI_PAPER_SIZE_ID)
    public int papersizeID;

    public CLSSPaperSizeInfo() {
        init();
    }

    private void init() {
        int[] iArr = new int[4];
        for (int i9 = 0; i9 < 4; i9++) {
            iArr[0] = 0;
        }
        set(65535, 65535, 65535, 65535, 65535, 65535, 65535, iArr, iArr, iArr, null);
    }

    public int[] getAvailableinputbin() {
        return this.availableinputbin;
    }

    public int getBorderlessprintLength() {
        return this.borderlessprintLength;
    }

    public int getBorderlessprintWidth() {
        return this.borderlessprintWidth;
    }

    public int getBorderprintLength() {
        return this.borderprintLength;
    }

    public int getBorderprintWidth() {
        return this.borderprintWidth;
    }

    public int getDuplexprintLength() {
        return this.duplexprintLength;
    }

    public int getDuplexprintWidth() {
        return this.duplexprintWidth;
    }

    public int[] getMarginBorder() {
        return this.marginBorder;
    }

    public int[] getMarginBorderless() {
        return this.marginBorderless;
    }

    public int[] getMarginDuplex() {
        return this.marginDuplex;
    }

    public int getPapersizeID() {
        return this.papersizeID;
    }

    public void set(int i9, int i10, int i11, int i12, int i13, int i14, int i15, int[] iArr, int[] iArr2, int[] iArr3, int[] iArr4) {
        this.papersizeID = i9;
        this.borderprintWidth = i10;
        this.borderprintLength = i11;
        this.borderlessprintWidth = i12;
        this.borderlessprintLength = i13;
        this.duplexprintWidth = i14;
        this.duplexprintLength = i15;
        for (int i16 = 0; i16 < 4; i16++) {
            this.marginBorder[i16] = iArr[i16];
            this.marginBorderless[i16] = iArr2[i16];
            this.marginDuplex[i16] = iArr3[i16];
        }
        this.availableinputbin = iArr4;
    }

    public void setAvailableinputbin(int[] iArr) {
        this.availableinputbin = iArr;
    }

    public void setBorderlessprintLength(int i9) {
        this.borderlessprintLength = i9;
    }

    public void setBorderlessprintWidth(int i9) {
        this.borderlessprintWidth = i9;
    }

    public void setBorderprintLength(int i9) {
        this.borderprintLength = i9;
    }

    public void setBorderprintWidth(int i9) {
        this.borderprintWidth = i9;
    }

    public void setDuplexprintLength(int i9) {
        this.duplexprintLength = i9;
    }

    public void setDuplexprintWidth(int i9) {
        this.duplexprintWidth = i9;
    }

    public void setMarginBorder(int[] iArr) {
        this.marginBorder = iArr;
    }

    public void setMarginBorderless(int[] iArr) {
        this.marginBorderless = iArr;
    }

    public void setMarginDuplex(int[] iArr) {
        this.marginDuplex = iArr;
    }

    public void setPapersizeID(int i9) {
        this.papersizeID = i9;
    }
}
