package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSStatusDeviceInfo extends CLSSStatusInfo {
    public String bluetoothAddress;
    public String eidRedirectUrl;
    public String eidUrl;
    public String eidUrlId;
    private boolean hasRootTag;
    private boolean hasSetupOnPrinterRootTag;
    private int mArsContractStatus;
    private String[] mColors;
    private int mCpcContractStatus;
    private String mCurrentSupportCode;
    private int mExtendedWarrantyStatus;
    private String[] mInkStatus;
    private int mInkSubscriptionStatus;
    private String mMainStatus;
    private int[] mOrders;
    private int mPrinterSubscriptionStatus;
    private int mProcessingRemain;
    private int mProcessingTotal;
    private int mProcessingUnit;
    private String mSetupOnPrinterCurrentSupportCode;
    private String mSetupOnPrinterMainStatus;
    private String[] mSupportParams;
    private int mTotalPaperNum;
    public String macAddressAPmode;
    public String macAddressWFD;
    public String macAddressWired;
    public String macAddressWireless;

    public CLSSStatusDeviceInfo() {
        init();
    }

    private boolean canCreateInkInfo() {
        String[] strArr;
        String[] strArr2;
        int[] iArr = this.mOrders;
        return (iArr == null || (strArr = this.mColors) == null || (strArr2 = this.mInkStatus) == null || iArr.length != strArr.length || iArr.length != strArr2.length) ? false : true;
    }

    public int getArsContractStatus() {
        return this.mArsContractStatus;
    }

    public int getCpcContractStatus() {
        return this.mCpcContractStatus;
    }

    public CLSSEidInfo getEidInfo() {
        CLSSEidInfo cLSSEidInfo = new CLSSEidInfo();
        cLSSEidInfo.set(this.eidUrlId, this.eidUrl, this.eidRedirectUrl);
        return cLSSEidInfo;
    }

    public int getExtendedWarrantyStatus() {
        return this.mExtendedWarrantyStatus;
    }

    public int getInkSubscriptionStatus() {
        return this.mInkSubscriptionStatus;
    }

    public int getPrinterSubscriptionStatus() {
        return this.mPrinterSubscriptionStatus;
    }

    public CLSSSetupInfo getSetupInfo() {
        CLSSDeviceInkInfo[] cLSSDeviceInkInfoArr;
        if (canCreateInkInfo()) {
            int[] iArr = this.mOrders;
            cLSSDeviceInkInfoArr = new CLSSDeviceInkInfo[iArr.length];
            int i9 = 0;
            for (int i10 : iArr) {
                cLSSDeviceInkInfoArr[i9] = new CLSSDeviceInkInfo(this.mColors[i9], this.mInkStatus[i9], i10);
                i9++;
            }
        } else {
            cLSSDeviceInkInfoArr = null;
        }
        return new CLSSSetupInfo(this.hasRootTag, this.mMainStatus, this.mSupportParams, this.mCurrentSupportCode, cLSSDeviceInkInfoArr, new CLSSDeviceProcessingStatus(this.mProcessingTotal, this.mProcessingRemain, this.mProcessingUnit));
    }

    public CLSSSetupOnPrinterInfo getSetupOnPrinterInfo() {
        return new CLSSSetupOnPrinterInfo(this.hasSetupOnPrinterRootTag, this.mSetupOnPrinterMainStatus, this.mSetupOnPrinterCurrentSupportCode);
    }

    public int getTotalPaperNum() {
        return this.mTotalPaperNum;
    }

    @Override // jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSStatusInfo
    public void init() {
        set(65535, 65535, null, 65535, null, null, null, null, null, null, null, null, null, false, null, null, null, null, null, null, -1, -1, -1, false, null, null, 65535, 65535, 65535, -1, 65535, 65535);
    }

    public void set(int i9, int i10, String str, int i11, String str2, String str3, String str4, String str5, String str6, String str7, String str8, String str9, String str10, boolean z9, String str11, String[] strArr, String str12, String[] strArr2, String[] strArr3, int[] iArr, int i12, int i13, int i14, boolean z10, String str13, String str14, int i15, int i16, int i17, int i18, int i19, int i20) {
        super.set(i9, i10, str, i11, str2);
        this.macAddressWired = newString(str3);
        this.macAddressWireless = newString(str4);
        this.macAddressAPmode = newString(str5);
        this.macAddressWFD = newString(str6);
        this.bluetoothAddress = newString(str7);
        this.eidUrlId = newString(str8);
        this.eidUrl = newString(str9);
        this.eidRedirectUrl = newString(str10);
        this.hasRootTag = z9;
        this.mMainStatus = str11;
        this.mSupportParams = strArr;
        this.mCurrentSupportCode = str12;
        this.mColors = strArr2;
        this.mInkStatus = strArr3;
        this.mOrders = iArr;
        this.mProcessingTotal = i12;
        this.mProcessingRemain = i13;
        this.mProcessingUnit = i14;
        this.hasSetupOnPrinterRootTag = z10;
        this.mSetupOnPrinterMainStatus = str13;
        this.mSetupOnPrinterCurrentSupportCode = str14;
        this.mPrinterSubscriptionStatus = i15;
        this.mArsContractStatus = i16;
        this.mInkSubscriptionStatus = i17;
        this.mTotalPaperNum = i18;
        this.mCpcContractStatus = i19;
        this.mExtendedWarrantyStatus = i20;
    }
}
