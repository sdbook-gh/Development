package k7;

import jp.co.canon.bsd.ad.sdk.core.util.struct.CipherData;

/**
 * Stub CipherHelper — serial decrypt/encrypt not needed for simplex CLSS print.
 */
public final class C1791a {
    private C1791a() {
    }

    public static String a(CipherData cipherData, String unusedKey, String unusedAgreement) {
        return "";
    }

    public static CipherData b(String plaintext, String unusedKey, String unusedAgreement) {
        byte[] bytes = plaintext == null ? new byte[0] : plaintext.getBytes();
        return new CipherData(bytes, new byte[0]);
    }
}
