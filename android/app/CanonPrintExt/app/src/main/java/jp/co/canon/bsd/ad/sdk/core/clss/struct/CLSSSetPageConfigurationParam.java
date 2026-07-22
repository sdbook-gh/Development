package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSSetPageConfigurationParam {
    public String jobID;
    public int nextpage;
    public int preparation;

    public CLSSSetPageConfigurationParam() {
        init();
    }

    public String getJobID() {
        return this.jobID;
    }

    public int getNextpage() {
        return this.nextpage;
    }

    public int getPreparation() {
        return this.preparation;
    }

    public void init() {
        set(null, 65535, 65535);
    }

    public void set(String str, int i9, int i10) {
        this.jobID = str;
        this.nextpage = i9;
        this.preparation = i10;
    }

    public void setNextPage(int i9) {
        this.nextpage = i9;
    }

    public void setPreraration(int i9) {
        this.preparation = i9;
    }

    public void setjobID(String str) {
        this.jobID = str;
    }
}
