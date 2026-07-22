package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import d7.InterfaceC1549a;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSSetJobCapabilityInfo {
    private static final String PREF_SEJCI_JOBCOPIES = "_sejci_jobcopies";
    private static final String PREF_SEJCI_MISMATCH_MODE = "_sejci_mismatch_mode";
    private static final int RANGE_MAX = 2;

    @InterfaceC1549a(key = PREF_SEJCI_JOBCOPIES)
    public int[] jobcopies = new int[2];

    @InterfaceC1549a(key = PREF_SEJCI_MISMATCH_MODE)
    public int[] mismatch_mode;
    public int version;

    public CLSSSetJobCapabilityInfo() {
        init();
    }

    private void init() {
        set(65535, new int[]{65535, 65535}, null);
    }

    public int[] getJobcopies() {
        return this.jobcopies;
    }

    public int[] getMismatch_mode() {
        return this.mismatch_mode;
    }

    public void set(int i9, int[] iArr, int[] iArr2) {
        this.version = i9;
        this.jobcopies = iArr;
        this.mismatch_mode = iArr2;
    }

    public void setJobcopies(int[] iArr) {
        this.jobcopies = iArr;
    }

    public void setMismatch_mode(int[] iArr) {
        this.mismatch_mode = iArr;
    }
}
