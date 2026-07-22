package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSSetupOnPrinterInfo {
    private String mCurrentSupportCode;
    private boolean mHasRootTag;
    private String mMainStatus;

    public CLSSSetupOnPrinterInfo() {
        set(false, null, null);
    }

    public String getCurrentSupportCode() {
        return this.mCurrentSupportCode;
    }

    public String getMainStatus() {
        return this.mMainStatus;
    }

    public boolean hasRootTag() {
        return this.mHasRootTag;
    }

    public void set(boolean z9, String str, String str2) {
        this.mHasRootTag = z9;
        this.mMainStatus = str;
        this.mCurrentSupportCode = str2;
    }

    public CLSSSetupOnPrinterInfo(boolean z9, String str, String str2) {
        this.mHasRootTag = z9;
        this.mMainStatus = str;
        this.mCurrentSupportCode = str2;
    }
}
