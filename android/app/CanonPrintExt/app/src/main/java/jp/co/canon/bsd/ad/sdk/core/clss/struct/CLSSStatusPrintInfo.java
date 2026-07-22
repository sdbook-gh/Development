package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSStatusPrintInfo extends CLSSStatusInfo {
    public int batteryInfo_level;
    public int[] batteryInfo_status;
    public long[] bininfo_current_papersize_width;
    public int[] bininfo_number;
    public int[] bininfo_papersizeID;
    public int[] bininfo_papertypeID;
    public int complete_impression;
    public int[] confirm_media;
    public String[] errorCode;
    public boolean flg_tca;
    public String[] inkinfo_color;
    public int[] inkinfo_inkstatus;
    public int[] inkinfo_level;
    public String[] inkinfo_model;
    public int[] inkinfo_order;
    public int[] itemId;
    public int listNum;
    public int[] severity;
    public int state;
    public int[] suffixCode;
    public int[] summary;

    public CLSSStatusPrintInfo() {
        init();
    }

    @Override // jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSStatusInfo
    public void init() {
        set(65535, 65535, null, 65535, null, null, null, null, false, 65535, 65535, null, null, null, null, null, null, null, 65535, null, null, null, null, null, null, 65535);
    }

    public void set(int i9, int i10, String str, int i11, String str2, int[] iArr, int[] iArr2, int[] iArr3, boolean z9, int i12, int i13, long[] jArr, int[] iArr4, int[] iArr5, int[] iArr6, int[] iArr7, String[] strArr, int[] iArr8, int i14, String[] strArr2, String[] strArr3, int[] iArr9, int[] iArr10, int[] iArr11, int[] iArr12, int i15) {
        super.set(i9, i10, str, i11, str2);
        if (iArr == null) {
            this.bininfo_papertypeID = null;
        } else {
            this.bininfo_papertypeID = new int[iArr.length];
            for (int i16 = 0; i16 < iArr.length; i16++) {
                this.bininfo_papertypeID[i16] = iArr[i16];
            }
        }
        if (iArr2 == null) {
            this.bininfo_papersizeID = null;
        } else {
            this.bininfo_papersizeID = new int[iArr2.length];
            for (int i17 = 0; i17 < iArr2.length; i17++) {
                this.bininfo_papersizeID[i17] = iArr2[i17];
            }
        }
        if (iArr3 == null) {
            this.confirm_media = null;
        } else {
            this.confirm_media = new int[iArr3.length];
            for (int i18 = 0; i18 < iArr3.length; i18++) {
                this.confirm_media[i18] = iArr3[i18];
            }
        }
        this.flg_tca = z9;
        this.complete_impression = i12;
        this.state = i13;
        this.bininfo_current_papersize_width = jArr;
        this.bininfo_number = iArr4;
        this.itemId = iArr5;
        this.severity = iArr6;
        this.summary = iArr7;
        this.errorCode = strArr;
        this.listNum = i14;
        this.suffixCode = iArr8;
        this.inkinfo_model = strArr2;
        this.inkinfo_color = strArr3;
        this.inkinfo_inkstatus = iArr9;
        this.inkinfo_level = iArr10;
        this.inkinfo_order = iArr11;
        this.batteryInfo_status = iArr12;
        this.batteryInfo_level = i15;
    }
}
