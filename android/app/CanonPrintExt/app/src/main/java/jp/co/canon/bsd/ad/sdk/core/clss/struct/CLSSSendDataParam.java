package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSSendDataParam {
    long dataLength;
    long dataSize;
    long dataWidth;
    int format;
    int isContinue;
    String jobID;

    public CLSSSendDataParam() {
        set(null, 65535, 4294967295L, 4294967295L, 4294967295L, 65535);
    }

    public void set(String str, int i9, long j9, long j10, long j11, int i10) {
        this.jobID = str;
        this.format = i9;
        this.dataSize = j9;
        this.dataWidth = j10;
        this.dataLength = j11;
        this.isContinue = i10;
    }

    public void setDataLength(long j9) {
        this.dataLength = j9;
    }

    public void setDataSize(long j9) {
        this.dataSize = j9;
    }

    public void setDataWidth(long j9) {
        this.dataWidth = j9;
    }

    public void setFormat(int i9) {
        this.format = i9;
    }

    public void setIsContinue(int i9) {
        this.isContinue = i9;
    }

    public void setJobID(String str) {
        this.jobID = str;
    }
}
