package k7;

import java.util.Arrays;
import java.util.Locale;
import j7.C1704a;

/** Util helpers used by SnmpSearch / BjnpSearch / BjnpSocket / CLSS workers. */
public final class h {
    private h() {
    }

    private static String hexByte(byte b) {
        return String.format(Locale.US, "%02X", b & 255);
    }

    public static void o() {
        p(200);
    }

    public static void p(int millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException unused) {
            Thread.currentThread().interrupt();
        }
    }

    public static byte[] f(String str) {
        if (str == null) {
            return null;
        }
        return str.getBytes(C1704a.f14726a);
    }

    public static byte[] b(byte[] existing, byte[] chunk, int len) {
        if (chunk == null || len <= 0) {
            return existing;
        }
        if (existing == null) {
            return Arrays.copyOf(chunk, len);
        }
        byte[] out = new byte[existing.length + len];
        System.arraycopy(existing, 0, out, 0, existing.length);
        System.arraycopy(chunk, 0, out, existing.length, len);
        return out;
    }

    /** Returns suffix of haystack starting at first match of needle, or null. */
    public static byte[] e(byte[] needle, byte[] haystack, int needleLen) {
        if (needle == null || haystack == null || needleLen <= 0 || haystack.length < needleLen) {
            return null;
        }
        outer:
        for (int i = 0; i <= haystack.length - needleLen; i++) {
            for (int j = 0; j < needleLen; j++) {
                if (haystack[i + j] != needle[j]) {
                    continue outer;
                }
            }
            return Arrays.copyOfRange(haystack, i, haystack.length);
        }
        return null;
    }

    /** IPv4 dotted string from 4 bytes. */
    public static String g(byte[] bArr) {
        if (bArr == null || bArr.length != 4) {
            return null;
        }
        return (bArr[0] & 255) + "." + (bArr[1] & 255) + "." + (bArr[2] & 255) + "." + (bArr[3] & 255);
    }

    /** MAC colon-hex string from 6 bytes. */
    public static String h(byte[] bArr) {
        if (bArr == null || bArr.length != 6) {
            return null;
        }
        return hexByte(bArr[0]) + ":" + hexByte(bArr[1]) + ":" + hexByte(bArr[2])
                + ":" + hexByte(bArr[3]) + ":" + hexByte(bArr[4]) + ":" + hexByte(bArr[5]);
    }

    /** IEEE1284 key value, e.g. m(deviceId, "MDL"). */
    public static String m(String deviceId, String key) {
        if (deviceId == null || key == null) {
            return null;
        }
        String[] parts = deviceId.split(";");
        String prefix = key + ":";
        for (String part : parts) {
            if (part.startsWith(prefix)) {
                return part.substring(prefix.length());
            }
        }
        return null;
    }

    /** True if deviceId key's comma-list contains value (e.g. MFG/Canon, CMD/IVEC). */
    public static boolean n(String deviceId, String key, String value) {
        String raw = m(deviceId, key);
        if (raw == null) {
            return false;
        }
        String[] split = raw.split(",");
        for (String s : split) {
            if (s.equals(value)) {
                return true;
            }
        }
        return false;
    }
}
