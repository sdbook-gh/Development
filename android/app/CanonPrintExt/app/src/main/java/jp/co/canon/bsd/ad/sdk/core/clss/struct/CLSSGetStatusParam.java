package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSGetStatusParam {
    String jobID;
    int serviceType;

    public CLSSGetStatusParam() {
        init();
    }

    public void init() {
        set(65535, null);
    }

    public void set(int i9, String str) {
        this.serviceType = i9;
        this.jobID = str;
    }

    public void setJobID(String str) {
        this.jobID = str;
    }

    public void setServiceType(int i9) {
        this.serviceType = i9;
    }
}
