package jp.co.canon.oip.android.opal.mobileatp.util;

/* compiled from: StringUtil.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class g {

    /* renamed from: a, reason: collision with root package name */
    public static final String f16947a = "";

    private g() {
    }

    public static boolean a(String str) {
        return b(str);
    }

    public static boolean b(String str) {
        return str == null || str.length() < 1;
    }

    public static String c(String str) {
        if (a(str)) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        int length = str.length();
        for (int i9 = 0; i9 < length; i9++) {
            char charAt = str.charAt(i9);
            if (charAt == '_' || (('0' <= charAt && charAt <= '9') || (('a' <= charAt && charAt <= 'z') || ('A' <= charAt && charAt <= 'Z')))) {
                sb.append(charAt);
            } else {
                String hexString = Integer.toHexString(charAt);
                int length2 = hexString.length();
                sb.append("\\u");
                for (int i10 = 0; i10 < 4 - length2; i10++) {
                    sb.append('0');
                }
                sb.append(hexString);
            }
        }
        return sb.toString();
    }

    public static int a(String str, char c10) {
        if (b(str)) {
            return 0;
        }
        int length = str.length();
        int i9 = 0;
        for (int i10 = 0; i10 < length; i10++) {
            if (str.charAt(i10) == c10) {
                i9++;
            }
        }
        return i9;
    }

    public static String[] b(String str, char c10) {
        try {
            String[] strArr = new String[a(str, c10) + 1];
            try {
                int length = str.length();
                StringBuilder sb = new StringBuilder();
                int i9 = 0;
                for (int i10 = 0; i10 < length; i10++) {
                    if (str.charAt(i10) != c10) {
                        sb.append(str.charAt(i10));
                    } else {
                        strArr[i9] = sb.toString();
                        sb = new StringBuilder();
                        i9++;
                    }
                }
                strArr[i9] = sb.toString();
                return strArr;
            } catch (OutOfMemoryError unused) {
                return strArr;
            }
        } catch (OutOfMemoryError unused2) {
            return null;
        }
    }

    public static char[] a(byte[] bArr) {
        try {
            char[] cArr = new char[bArr.length];
            try {
                int length = bArr.length;
                for (int i9 = 0; i9 < length; i9++) {
                    cArr[i9] = (char) bArr[i9];
                }
                return cArr;
            } catch (OutOfMemoryError unused) {
                return cArr;
            }
        } catch (OutOfMemoryError unused2) {
            return null;
        }
    }

    public static byte[] a(char[] cArr) {
        try {
            byte[] bArr = new byte[cArr.length];
            try {
                int length = cArr.length;
                for (int i9 = 0; i9 < length; i9++) {
                    bArr[i9] = (byte) cArr[i9];
                }
                return bArr;
            } catch (OutOfMemoryError unused) {
                return bArr;
            }
        } catch (OutOfMemoryError unused2) {
            return null;
        }
    }

    public static void b(byte[] bArr) {
        StringBuilder sb = new StringBuilder();
        int length = bArr.length;
        for (int i9 = 0; i9 < length; i9++) {
            if (i9 != 0 && i9 % 16 == 0) {
                sb.append('\n');
                jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, sb.toString());
                sb = new StringBuilder();
            }
            sb.append("0x");
            sb.append(String.format("%02x", Byte.valueOf(bArr[i9])));
            sb.append(',');
        }
        if (sb.length() > 0) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, sb.toString());
        }
    }

    public static void b(char[] cArr) {
        StringBuilder sb = new StringBuilder();
        for (char c10 : cArr) {
            sb.append(Integer.toHexString(c10));
        }
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, sb.toString());
    }
}
