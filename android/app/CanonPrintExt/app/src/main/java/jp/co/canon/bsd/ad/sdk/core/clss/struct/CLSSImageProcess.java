package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import d7.InterfaceC1549a;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSImageProcess {
    private static final String PREF_IP_HASIMAGECORRECTIONAUTO = "_ip_hasImagecorrectionAuto";
    private static final String PREF_IP_HASSHARPNESSAUTO = "_ip_hasSharpnessAuto";
    private static final String PREF_IP_IMAGECORRECTION = "_ip_imagecorrection";
    private static final String PREF_IP_PRINTUSE = "_ip_printUse";
    private static final String PREF_IP_SHARPNESS = "_ip_sharpness";

    @InterfaceC1549a(key = PREF_IP_HASIMAGECORRECTIONAUTO)
    public boolean hasImagecorrectionAuto;

    @InterfaceC1549a(key = PREF_IP_HASSHARPNESSAUTO)
    public boolean hasSharpnessAuto;

    @InterfaceC1549a(key = PREF_IP_IMAGECORRECTION)
    public int[] imagecorrection;

    @InterfaceC1549a(defInt = 65535, key = PREF_IP_PRINTUSE)
    public int printUse;

    @InterfaceC1549a(key = PREF_IP_SHARPNESS)
    public int[] sharpness;

    public CLSSImageProcess() {
        init();
    }

    public void init() {
        this.printUse = 65535;
        this.imagecorrection = new int[4];
        int i9 = 0;
        while (true) {
            int[] iArr = this.imagecorrection;
            if (i9 >= iArr.length) {
                break;
            }
            iArr[i9] = 65535;
            i9++;
        }
        this.hasImagecorrectionAuto = false;
        this.sharpness = new int[4];
        int i10 = 0;
        while (true) {
            int[] iArr2 = this.sharpness;
            if (i10 >= iArr2.length) {
                this.hasSharpnessAuto = false;
                return;
            } else {
                iArr2[i10] = 65535;
                i10++;
            }
        }
    }
}
