package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import d7.InterfaceC1549a;
import java.util.Objects;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSCorrespondPaperSize {
    private static final String PREF_CP_HEIGHT = "_cp_height";
    private static final String PREF_CP_ID = "_cp_id";
    private static final String PREF_CP_IS_PORTRAIT = "_cp_is_portrait";
    private static final String PREF_CP_WIDTH = "_cp_widht";

    @InterfaceC1549a(key = PREF_CP_HEIGHT)
    public int height;

    @InterfaceC1549a(defInt = 65535, key = PREF_CP_ID)
    public int id;

    @InterfaceC1549a(defBoolean = false, key = PREF_CP_IS_PORTRAIT)
    public boolean isPortrait;

    @InterfaceC1549a(key = PREF_CP_WIDTH)
    public int width;

    public CLSSCorrespondPaperSize() {
        init();
    }

    public boolean equals(Object obj) {
        return obj != null && (obj instanceof CLSSCorrespondPaperSize) && ((CLSSCorrespondPaperSize) obj).id == this.id;
    }

    public int hashCode() {
        return Objects.hash(
                Integer.valueOf(this.id),
                Boolean.valueOf(this.isPortrait),
                Integer.valueOf(this.width),
                Integer.valueOf(this.height));
    }

    public void init() {
        this.id = 65535;
        this.isPortrait = false;
        this.height = 0;
        this.width = 0;
    }
}
