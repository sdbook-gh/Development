package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSStatusInfo {
    public String jobID;
    public int jobprogress;
    public int status;
    public int statusDetail;
    public String support_codeID;

    public CLSSStatusInfo() {
        init();
    }

    public void init() {
        set(65535, 65535, null, 65535, null);
    }

    public String newString(String str) {
        if (str == null) {
            return null;
        }
        try {
            return new String(str);
        } catch (Exception e10) {
            e10.printStackTrace();
            return null;
        }
    }

    public void set(int i9, int i10, String str, int i11, String str2) {
        this.status = i9;
        this.statusDetail = i10;
        this.jobID = newString(str);
        this.jobprogress = i11;
        this.support_codeID = newString(str2);
    }
}
