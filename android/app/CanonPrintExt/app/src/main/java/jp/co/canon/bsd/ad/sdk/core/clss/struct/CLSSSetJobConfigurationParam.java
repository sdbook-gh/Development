package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSSetJobConfigurationParam {
    public String datetime;
    public int deviceSideGuide;
    public int jobCopies;
    public String jobID;
    public int mediaDetec;
    public int mismatchMode;

    public CLSSSetJobConfigurationParam() {
        init();
    }

    public void init() {
        set(null, null, 65535, 65535, 65535, 65535);
    }

    public void set(String str, String str2, int i9, int i10, int i11, int i12) {
        this.jobID = str;
        this.datetime = str2;
        this.deviceSideGuide = i9;
        this.mediaDetec = i10;
        this.jobCopies = i11;
        this.mismatchMode = i12;
    }

    public void setDateTime(String str) {
        this.datetime = str;
    }

    public void setDeviceSideGuide(int i9) {
        this.deviceSideGuide = i9;
    }

    public void setJobCopies(int i9) {
        this.jobCopies = i9;
    }

    public void setMediaDetec(int i9) {
        this.mediaDetec = i9;
    }

    public void setMismatchMode(int i9) {
        this.mismatchMode = i9;
    }

    public void setjobID(String str) {
        this.jobID = str;
    }
}
