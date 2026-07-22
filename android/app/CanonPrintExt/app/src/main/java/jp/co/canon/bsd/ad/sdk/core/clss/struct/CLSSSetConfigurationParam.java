package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSSetConfigurationParam {
    String jobID;
    public int serviceType;
    public CLSSPrintSettingsInfo printSettings = new CLSSPrintSettingsInfo();
    public CLSSConfigurationDeviceInfo deviceSettings = new CLSSConfigurationDeviceInfo();

    public CLSSSetConfigurationParam() {
        this.printSettings.init();
        this.deviceSettings.init();
        init();
    }

    public CLSSConfigurationDeviceInfo getDeviceSettings() {
        return this.deviceSettings;
    }

    public CLSSPrintSettingsInfo getPrintSettings() {
        return this.printSettings;
    }

    public void init() {
        this.printSettings.init();
        this.deviceSettings.init();
        setServiceType(65535);
        setJobID(null);
    }

    public void set(int i9, CLSSPrintSettingsInfo cLSSPrintSettingsInfo, CLSSConfigurationDeviceInfo cLSSConfigurationDeviceInfo) {
        this.serviceType = i9;
        this.printSettings = cLSSPrintSettingsInfo;
        this.deviceSettings = cLSSConfigurationDeviceInfo;
    }

    public void setDeviceSettings(CLSSConfigurationDeviceInfo cLSSConfigurationDeviceInfo) {
        this.deviceSettings = cLSSConfigurationDeviceInfo;
    }

    public void setJobID(String str) {
        this.jobID = str;
    }

    public void setPrintSettings(CLSSPrintSettingsInfo cLSSPrintSettingsInfo) {
        this.printSettings = cLSSPrintSettingsInfo;
    }

    public void setServiceType(int i9) {
        this.serviceType = i9;
    }
}
