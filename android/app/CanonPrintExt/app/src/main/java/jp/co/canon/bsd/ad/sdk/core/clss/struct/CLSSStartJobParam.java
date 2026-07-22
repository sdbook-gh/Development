package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSStartJobParam {
    public int applicationID;
    public String bidi;
    public int box;
    public String computerName;
    public int forcepmdetection;
    public int hostEnvID;
    public String jkey1;
    public int jobExecutionMode;
    public int jobExecutionTiming;
    public String jobID;
    public String jobName;
    public int keyMisdetection;
    public int serviceType;
    public String userName;
    public String uuid;

    public CLSSStartJobParam() {
        set(null, null, 65535, 65535, 65535, 65535, null, 65535, 65535, 65535, null, null, null, null, 65535);
    }

    public void set(String str, String str2, int i9, int i10, int i11, int i12, String str3, int i13, int i14, int i15, String str4, String str5, String str6, String str7, int i16) {
        this.jobID = str;
        this.bidi = str2;
        this.keyMisdetection = i9;
        this.forcepmdetection = i10;
        this.serviceType = i11;
        this.hostEnvID = i12;
        this.uuid = str3;
        this.jobExecutionMode = i13;
        this.jobExecutionTiming = i14;
        this.box = i15;
        this.jobName = str4;
        this.userName = str5;
        this.computerName = str6;
        this.jkey1 = str7;
        this.applicationID = i16;
    }

    public void setApplicationID(int i9) {
        this.applicationID = i9;
    }

    public void setBidi(String str) {
        this.bidi = str;
    }

    public void setBox(int i9) {
        this.box = i9;
    }

    public void setComputerName(String str) {
        this.computerName = str;
    }

    public void setForcepmdetection(int i9) {
        this.forcepmdetection = i9;
    }

    public void setHostEnvID(int i9) {
        this.hostEnvID = i9;
    }

    public void setJkey1(String str) {
        this.jkey1 = str;
    }

    public void setJobExecutionMode(int i9) {
        this.jobExecutionMode = i9;
    }

    public void setJobExecutionTiming(int i9) {
        this.jobExecutionTiming = i9;
    }

    public void setJobID(String str) {
        this.jobID = str;
    }

    public void setJobName(String str) {
        this.jobName = str;
    }

    public void setKeyMisdetection(int i9) {
        this.keyMisdetection = i9;
    }

    public void setServiceType(int i9) {
        this.serviceType = i9;
    }

    public void setUserName(String str) {
        this.userName = str;
    }

    public void setUuid(String str) {
        this.uuid = str;
    }
}
