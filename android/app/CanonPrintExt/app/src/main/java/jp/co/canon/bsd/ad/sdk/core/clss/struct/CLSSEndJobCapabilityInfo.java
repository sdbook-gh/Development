package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import d7.InterfaceC1549a;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSEndJobCapabilityInfo {
    private static final String PREF_EJCI_FLG_IMPRESSION_NUM = "_ejci_flg_impression_num ";

    @InterfaceC1549a(key = PREF_EJCI_FLG_IMPRESSION_NUM)
    public boolean flg_impression_num;
    public int version;

    public CLSSEndJobCapabilityInfo() {
        init();
    }

    private void init() {
        set(65535, false);
    }

    public boolean isFlg_impression_num() {
        return this.flg_impression_num;
    }

    public void set(int i9, boolean z9) {
        this.version = i9;
        this.flg_impression_num = z9;
    }

    public void setFlg_impression_num(boolean z9) {
        this.flg_impression_num = z9;
    }
}
