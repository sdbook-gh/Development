package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import d7.InterfaceC1549a;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSSendDataCapabilityInfo {
    private static final String PREF_SDCI_FLG_CONTINUE = "_sdci_flg_continue";

    @InterfaceC1549a(key = PREF_SDCI_FLG_CONTINUE)
    public boolean flg_continue;
    public int version;

    public CLSSSendDataCapabilityInfo() {
        init();
    }

    private void init() {
        set(65535, false);
    }

    public boolean isFlg_continue() {
        return this.flg_continue;
    }

    public void set(int i9, boolean z9) {
        this.version = i9;
        this.flg_continue = z9;
    }

    public void setFlg_continue(boolean z9) {
        this.flg_continue = z9;
    }
}
