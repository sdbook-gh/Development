package jp.co.canon.bsd.ad.sdk.core.clss;

/**
 * Minimal CLSSUtility for discovery: protocol parse via libsdk-core JNI.
 */
public final class CLSSUtility {
    private static final String STR_ERROR =
            "load library Error( nothing code \" System.loadLibrary();\" or nothing JNI folder)";

    private CLSSUtility() {
    }

    private static native int WrapperCLSSGetProtocol(String deviceId);

    public static int getProtocol(String deviceId) {
        try {
            return WrapperCLSSGetProtocol(deviceId);
        } catch (Exception e) {
            throw new CLSS_Exception(e.toString(), -3);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(STR_ERROR);
        }
    }
}
