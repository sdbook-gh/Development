package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import d7.InterfaceC1549a;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSStartJobCapabilityInfo {
    private static final String PREF_SJCI_BOX = "_sjci_box";
    private static final String PREF_SJCI_FLG_COMPUTERNAME = "_sjci_flg_computername";
    private static final String PREF_SJCI_FLG_JKEY1 = "_sjci_flg_jkey1";
    private static final String PREF_SJCI_FLG_JOBNAME = "_sjci_flg_jobname";
    private static final String PREF_SJCI_FLG_JOB_DESCRIPTION = "_sjci_flg_job_description";
    private static final String PREF_SJCI_FLG_USERNAME = "_sjci_flg_username";
    private static final String PREF_SJCI_JOB_EXECUTION_MODE = "_sjci_job_execution_mode";
    private static final String PREF_SJCI_JOB_EXECUTION_TIMING = "_sjci_job_execution_timing";
    private static final int RANGE_MAX = 2;

    @InterfaceC1549a(key = PREF_SJCI_BOX)
    public int[] box = new int[2];

    @InterfaceC1549a(key = PREF_SJCI_FLG_COMPUTERNAME)
    public boolean flg_computername;

    @InterfaceC1549a(key = PREF_SJCI_FLG_JKEY1)
    public boolean flg_jkey1;

    @InterfaceC1549a(key = PREF_SJCI_FLG_JOB_DESCRIPTION)
    public boolean flg_job_description;

    @InterfaceC1549a(key = PREF_SJCI_FLG_JOBNAME)
    public boolean flg_jobname;

    @InterfaceC1549a(key = PREF_SJCI_FLG_USERNAME)
    public boolean flg_username;

    @InterfaceC1549a(key = PREF_SJCI_JOB_EXECUTION_MODE)
    public int[] job_execution_mode;

    @InterfaceC1549a(key = PREF_SJCI_JOB_EXECUTION_TIMING)
    public int[] job_execution_timing;
    public int version;

    public CLSSStartJobCapabilityInfo() {
        init();
    }

    private void init() {
        set(65535, null, null, new int[]{65535, 65535}, false, false, false, false, false);
    }

    public int[] getBox() {
        return this.box;
    }

    public int[] getJob_execution_mode() {
        return this.job_execution_mode;
    }

    public int[] getJob_execution_timing() {
        return this.job_execution_timing;
    }

    public boolean isFlg_computername() {
        return this.flg_computername;
    }

    public boolean isFlg_jkey1() {
        return this.flg_jkey1;
    }

    public boolean isFlg_job_description() {
        return this.flg_job_description;
    }

    public boolean isFlg_jobname() {
        return this.flg_jobname;
    }

    public boolean isFlg_username() {
        return this.flg_username;
    }

    public void set(int i9, int[] iArr, int[] iArr2, int[] iArr3, boolean z9, boolean z10, boolean z11, boolean z12, boolean z13) {
        this.version = i9;
        this.job_execution_mode = iArr;
        this.job_execution_timing = iArr2;
        this.box = iArr3;
        this.flg_jobname = z9;
        this.flg_username = z10;
        this.flg_computername = z11;
        this.flg_job_description = z12;
        this.flg_jkey1 = z13;
    }

    public void setBox(int[] iArr) {
        this.box = iArr;
    }

    public void setFlg_computername(boolean z9) {
        this.flg_computername = z9;
    }

    public void setFlg_jkey1(boolean z9) {
        this.flg_jkey1 = z9;
    }

    public void setFlg_job_description(boolean z9) {
        this.flg_job_description = z9;
    }

    public void setFlg_jobname(boolean z9) {
        this.flg_jobname = z9;
    }

    public void setFlg_username(boolean z9) {
        this.flg_username = z9;
    }

    public void setJob_execution_mode(int[] iArr) {
        this.job_execution_mode = iArr;
    }

    public void setJob_execution_timing(int[] iArr) {
        this.job_execution_timing = iArr;
    }
}
