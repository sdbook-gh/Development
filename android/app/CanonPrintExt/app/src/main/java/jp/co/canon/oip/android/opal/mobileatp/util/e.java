package jp.co.canon.oip.android.opal.mobileatp.util;

/* compiled from: Hex.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class e {
    public static String a(String str) {
        byte[] bArr = null;
        try {
            int length = str.length() / 2;
            bArr = new byte[length];
            int i9 = 0;
            while (i9 < length) {
                int i10 = i9 + 1;
                bArr[i9] = (byte) Integer.parseInt(str.substring(i9 * 2, i10 * 2), 16);
                i9 = i10;
            }
        } catch (OutOfMemoryError e10) {
            e10.printStackTrace();
        }
        return new String(bArr);
    }
}
