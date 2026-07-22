package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSResponseInfo {
    public int ijoperationID;
    public int ijresponse;
    public String jobID;
    public int operationID;
    public int response;
    public int responseDetail;
    public int serviceType;

    public CLSSResponseInfo() {
        init();
    }

    private String newString(String str) {
        if (str == null) {
            return null;
        }
        try {
            return new String(str);
        } catch (Exception unused) {
            return null;
        }
    }

    public void init() {
        set(65535, 65535, 65535, 65535, 65535, 65535, null);
    }

    public void set(int i9, int i10, int i11, int i12, int i13, int i14, String str) {
        this.serviceType = i9;
        this.operationID = i10;
        this.response = i11;
        this.responseDetail = i12;
        this.ijoperationID = i13;
        this.ijresponse = i14;
        this.jobID = newString(str);
    }
}
