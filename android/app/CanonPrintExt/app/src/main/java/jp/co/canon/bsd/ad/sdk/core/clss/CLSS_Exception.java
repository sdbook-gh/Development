package jp.co.canon.bsd.ad.sdk.core.clss;

public class CLSS_Exception extends RuntimeException {
    public static final int CLSS_ERROR_OTHER = -3;
    public int rtn;

    public CLSS_Exception() {
        super("");
        this.rtn = CLSS_ERROR_OTHER;
    }

    public CLSS_Exception(String message) {
        super(message);
        this.rtn = CLSS_ERROR_OTHER;
    }

    public CLSS_Exception(String message, int code) {
        super(message);
        this.rtn = code;
    }
}
