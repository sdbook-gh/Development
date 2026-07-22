package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSSetupInfo {
    private boolean hasRootTag;
    private String mCurrentSupportCode;
    private CLSSDeviceInkInfo[] mDeviceInkInfo;
    private String mMainStatus;
    private CLSSDeviceProcessingStatus mProcessingStatus;
    private String[] mSupportParams;

    public CLSSSetupInfo() {
        set(false, null, null, null, null, null);
    }

    public String getCurrentSupportCode() {
        return this.mCurrentSupportCode;
    }

    public CLSSDeviceInkInfo[] getDeviceInkInfo() {
        return this.mDeviceInkInfo;
    }

    public String getMainStatus() {
        return this.mMainStatus;
    }

    public CLSSDeviceProcessingStatus getProcessingStatus() {
        return this.mProcessingStatus;
    }

    public String[] getSupportParams() {
        return this.mSupportParams;
    }

    public boolean hasRootTag() {
        return this.hasRootTag;
    }

    public void set(boolean z9, String str, String[] strArr, String str2, CLSSDeviceInkInfo[] cLSSDeviceInkInfoArr, CLSSDeviceProcessingStatus cLSSDeviceProcessingStatus) {
        this.hasRootTag = z9;
        this.mMainStatus = str;
        this.mSupportParams = strArr;
        this.mCurrentSupportCode = str2;
        this.mDeviceInkInfo = cLSSDeviceInkInfoArr;
        this.mProcessingStatus = cLSSDeviceProcessingStatus;
    }

    public CLSSSetupInfo(boolean z9, String str, String[] strArr, String str2, CLSSDeviceInkInfo[] cLSSDeviceInkInfoArr, CLSSDeviceProcessingStatus cLSSDeviceProcessingStatus) {
        this.hasRootTag = z9;
        this.mMainStatus = str;
        this.mSupportParams = strArr;
        this.mCurrentSupportCode = str2;
        this.mDeviceInkInfo = cLSSDeviceInkInfoArr;
        this.mProcessingStatus = cLSSDeviceProcessingStatus;
    }
}
