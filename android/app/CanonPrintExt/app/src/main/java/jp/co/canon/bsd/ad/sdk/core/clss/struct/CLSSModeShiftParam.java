package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSModeShiftParam {
    String ijmode;
    String jobID;
    int serviceType;

    public CLSSModeShiftParam() {
        init();
    }

    public void init() {
        set(65535, null, null);
    }

    public void set(int i9, String str, String str2) {
        this.serviceType = i9;
        this.jobID = str;
        this.ijmode = str2;
    }

    public void setIjMode(String str) {
        this.ijmode = str;
    }

    public void setJobID(String str) {
        this.jobID = str;
    }

    public void setServiceType(int i9) {
        this.serviceType = i9;
    }
}
