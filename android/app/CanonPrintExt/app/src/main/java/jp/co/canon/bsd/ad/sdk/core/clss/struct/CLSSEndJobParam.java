package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSEndJobParam {
    public long impressionNum;
    public String jobID;
    public int serviceType;

    public CLSSEndJobParam() {
        init();
    }

    public void init() {
        set(65535, null, 4294967295L);
    }

    public void set(int i9, String str, long j9) {
        this.serviceType = i9;
        this.jobID = str;
        this.impressionNum = j9;
    }

    public void setImpressionNum(long j9) {
        this.impressionNum = j9;
    }

    public void setJobID(String str) {
        this.jobID = str;
    }

    public void setServiceType(int i9) {
        this.serviceType = i9;
    }
}
