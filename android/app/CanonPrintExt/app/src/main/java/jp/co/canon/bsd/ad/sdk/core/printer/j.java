package jp.co.canon.bsd.ad.sdk.core.printer;

import jp.co.canon.bsd.ad.sdk.core.clss.CLSSUtility;
import jp.co.canon.bsd.ad.sdk.core.clss.CLSS_Exception;

/** Minimal protocol helper used by SnmpSearch.setPrinter. */
public final class j {
    private j() {
    }

    public static int c(String deviceId) {
        try {
            return CLSSUtility.getProtocol(deviceId);
        } catch (CLSS_Exception e) {
            e.toString();
            return 1;
        }
    }
}
