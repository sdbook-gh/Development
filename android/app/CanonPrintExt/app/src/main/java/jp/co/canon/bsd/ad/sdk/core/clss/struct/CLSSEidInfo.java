package jp.co.canon.bsd.ad.sdk.core.clss.struct;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSEidInfo {
    public String redirect_url;
    public String url;
    public String url_id;

    public CLSSEidInfo() {
        init();
    }

    public void init() {
        set("", "", "");
    }

    public void set(String str, String str2, String str3) {
        this.url_id = str;
        this.url = str2;
        this.redirect_url = str3;
    }
}
