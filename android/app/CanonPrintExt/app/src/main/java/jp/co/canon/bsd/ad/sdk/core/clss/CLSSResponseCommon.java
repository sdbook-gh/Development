package jp.co.canon.bsd.ad.sdk.core.clss;

import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSResponseInfo;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSResponseCommon extends CLSSResponseInfo {
    private String str_error = "load library Error( nothing code \" System.loadLibrary();\" or nothing JNI folder)";

    public CLSSResponseCommon(String str) {
        try {
            super.init();
            WrapperCLSSParseResponseCommon(str);
        } catch (Exception e10) {
            super.init();
            throw new CLSS_Exception(e10.toString());
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        }
    }

    public native int WrapperCLSSGetCommandOperationPair(int i9);

    public native int WrapperCLSSGetOperationPair(int i9);

    public native int WrapperCLSSParseResponseCommon(String str);

    public int getOperationPair() {
        int i9 = this.operationID;
        if (i9 == 65535) {
            return 65535;
        }
        return WrapperCLSSGetOperationPair(i9);
    }
}
