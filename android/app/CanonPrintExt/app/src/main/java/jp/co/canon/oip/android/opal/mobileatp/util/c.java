package jp.co.canon.oip.android.opal.mobileatp.util;

import android.util.Base64;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import jp.co.canon.oip.android.opal.mobileatp.ATPCAMSConnectSetting;
import jp.co.canon.oip.android.opal.mobileatp.ATPMobileATP;
import jp.co.canon.oip.android.opal.mobileatp.ATPProxySetting;
import jp.co.canon.oip.android.opal.mobileatp.ATPResult;
import jp.co.canon.oip.android.opal.mobileatp.deviceregistration.ATPDeviceRegistrationRequest;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import jp.co.canon.oip.android.opal.mobileatp.token.ATPResultAccessToken;

/* compiled from: CryptoUtility.java */
/* loaded from: /mnt/f/print/classes3.dex */
public final class c {

    /* renamed from: a, reason: collision with root package name */
    private static final int f16934a = 20;

    /* renamed from: b, reason: collision with root package name */
    private static final String f16935b = "canon";

    /* renamed from: c, reason: collision with root package name */
    private static final int f16936c = 256;

    /* renamed from: d, reason: collision with root package name */
    private static final int f16937d = 0;

    /* renamed from: e, reason: collision with root package name */
    static final int f16938e = 1;

    /* renamed from: f, reason: collision with root package name */
    private static final Object f16939f = new Object();

    /* renamed from: g, reason: collision with root package name */
    private static final char[] f16940g = {134, 156, 148, '\b', 162, 'f', 'Z', 254, 227, 166, 153, '^', 150, '3', '3', 190};

    /* renamed from: h, reason: collision with root package name */
    private static final String f16941h = "AES";

    /* renamed from: i, reason: collision with root package name */
    private static final String f16942i = "AES/CBC/PKCS5Padding";

    /* renamed from: j, reason: collision with root package name */
    private static final String f16943j = "MD5";

    /* renamed from: k, reason: collision with root package name */
    private static final String f16944k = "SHA1";

    /* renamed from: l, reason: collision with root package name */
    private static byte[] f16945l;

    /* compiled from: CryptoUtility.java */
    public static class a {

        /* renamed from: a, reason: collision with root package name */
        public static String f16946a;

        public static String a() {
            if (f16946a == null) {
                f16946a = c.a(1);
            }
            return f16946a;
        }
    }

    public static String a(String str, String str2) {
        synchronized (f16939f) {
            try {
                if (g.a(str2)) {
                    return "";
                }
                return c(str, str2);
            } catch (Throwable th) {
                throw th;
            }
        }
    }

    public static String b(String str, String str2) {
        synchronized (f16939f) {
            try {
                if (g.a(str2)) {
                    return "";
                }
                return e(str, str2);
            } catch (Throwable th) {
                throw th;
            }
        }
    }

    public static String c(String str) {
        synchronized (f16939f) {
            try {
                if (g.a(str)) {
                    return "";
                }
                String a10 = a.a();
                if (g.a(a10)) {
                    a10 = f16935b;
                }
                return c(a10, str);
            } catch (Throwable th) {
                throw th;
            }
        }
    }

    public static String d(String str) {
        synchronized (f16939f) {
            try {
                if (g.a(str)) {
                    return "";
                }
                String a10 = a.a();
                if (g.a(a10)) {
                    a10 = f16935b;
                }
                return d(a10, str);
            } catch (Throwable th) {
                throw th;
            }
        }
    }

    private static String e(String str, String str2) {
        try {
            byte[] a10 = a(a(str), str2.getBytes(), 1);
            byte[] e10 = e(str2);
            byte[] bArr = new byte[a10.length + e10.length];
            System.arraycopy(a10, 0, bArr, 0, a10.length);
            System.arraycopy(e10, 0, bArr, a10.length, e10.length);
            return new String(Base64.encode(bArr, 2));
        } catch (OutOfMemoryError e11) {
            throw new ATPException(400, e11.getMessage(), e11);
        } catch (ATPException e12) {
            throw e12;
        } catch (Exception e13) {
            throw new ATPException(400, e13.getMessage(), e13);
        }
    }

    public static String f(String str) {
        synchronized (f16939f) {
            try {
                if (g.a(str)) {
                    return "";
                }
                String a10 = a.a();
                if (g.a(a10)) {
                    a10 = f16935b;
                }
                return e(a10, str);
            } catch (Throwable th) {
                throw th;
            }
        }
    }

    public static String g(String str) {
        synchronized (f16939f) {
            try {
                if (g.a(str)) {
                    return "";
                }
                String a10 = a.a();
                if (g.a(a10)) {
                    a10 = f16935b;
                }
                return f(a10, str);
            } catch (Throwable th) {
                throw th;
            }
        }
    }

    private static byte[] a(String str) {
        try {
            MessageDigest messageDigest = MessageDigest.getInstance(f16943j);
            messageDigest.update(str.getBytes(), 0, str.getBytes().length);
            byte[] digest = messageDigest.digest();
            String d10 = jp.co.canon.oip.android.opal.mobileatp.f.b.g().h().d();
            messageDigest.update(d10.getBytes(), 0, d10.getBytes().length);
            byte[] digest2 = messageDigest.digest();
            byte[] bArr = new byte[digest.length + digest2.length];
            System.arraycopy(digest, 0, bArr, 0, digest.length);
            System.arraycopy(digest2, 0, bArr, digest.length, digest2.length);
            return bArr;
        } catch (OutOfMemoryError e10) {
            throw new ATPException(400, e10.getMessage(), e10);
        } catch (NoSuchAlgorithmException e11) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_CRYPTO_NO_SUCH_ALGORITHM, e11.getMessage(), e11);
        }
    }

    private static byte[] b(String str) {
        try {
            MessageDigest messageDigest = MessageDigest.getInstance(f16943j);
            messageDigest.update(str.getBytes(), 0, str.getBytes().length);
            byte[] digest = messageDigest.digest();
            byte[] bArr = new byte[digest.length];
            System.arraycopy(digest, 0, bArr, 0, digest.length);
            return bArr;
        } catch (OutOfMemoryError e10) {
            throw new ATPException(400, e10.getMessage(), e10);
        } catch (NoSuchAlgorithmException e11) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_CRYPTO_NO_SUCH_ALGORITHM, e11.getMessage(), e11);
        }
    }

    private static String c(String str, String str2) {
        try {
            byte[] a10 = a(str);
            byte[] decode = Base64.decode(str2.getBytes(), 2);
            byte[] bArr = new byte[decode.length - 20];
            byte[] bArr2 = new byte[20];
            System.arraycopy(decode, 0, bArr, 0, decode.length - 20);
            System.arraycopy(decode, decode.length - 20, bArr2, 0, 20);
            byte[] a11 = a(a10, bArr, 2);
            if (new String(e(new String(a11))).equals(new String(bArr2))) {
                return new String(a11);
            }
            throw new ATPException(ATPResult.RESULT_CODE_NG_DECRYPT_KEY, "invalid decrypt key. not much.");
        } catch (ATPException e12) {
            throw e12;
        } catch (Exception e10) {
            throw new ATPException(400, e10.getMessage(), e10);
        } catch (OutOfMemoryError e11) {
            throw new ATPException(400, e11.getMessage(), e11);
        }
    }

    private static String d(String str, String str2) {
        try {
            byte[] b10 = b(str);
            byte[] decode = Base64.decode(str2.getBytes(), 2);
            byte[] bArr = new byte[decode.length - 20];
            byte[] bArr2 = new byte[20];
            System.arraycopy(decode, 0, bArr, 0, decode.length - 20);
            System.arraycopy(decode, decode.length - 20, bArr2, 0, 20);
            byte[] a10 = a(b10, bArr, 2);
            if (new String(e(new String(a10))).equals(new String(bArr2))) {
                return new String(a10);
            }
            throw new ATPException(ATPResult.RESULT_CODE_NG_DECRYPT_KEY, "invalid decrypt key. not much.");
        } catch (ATPException e12) {
            throw e12;
        } catch (Exception e10) {
            throw new ATPException(400, e10.getMessage(), e10);
        } catch (OutOfMemoryError e11) {
            throw new ATPException(400, e11.getMessage(), e11);
        }
    }

    private static String f(String str, String str2) {
        try {
            byte[] a10 = a(b(str), str2.getBytes(), 1);
            byte[] e10 = e(str2);
            byte[] bArr = new byte[a10.length + e10.length];
            System.arraycopy(a10, 0, bArr, 0, a10.length);
            System.arraycopy(e10, 0, bArr, a10.length, e10.length);
            return new String(Base64.encode(bArr, 2));
        } catch (OutOfMemoryError e11) {
            throw new ATPException(400, e11.getMessage(), e11);
        } catch (ATPException e12) {
            throw e12;
        } catch (Exception e13) {
            throw new ATPException(400, e13.getMessage(), e13);
        }
    }

    public static byte[] e(String str) {
        try {
            MessageDigest messageDigest = MessageDigest.getInstance(f16944k);
            messageDigest.update(str.getBytes(), 0, str.getBytes().length);
            return messageDigest.digest();
        } catch (NoSuchAlgorithmException e10) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_CRYPTO_INVALID_ALGORITHM, e10.getMessage(), e10);
        }
    }

    public static byte[] b() {
        if (f16945l == null) {
            String a10 = a(0);
            byte[] bArr = {0};
            if (!g.a(a10)) {
                bArr = Base64.decode(e.a(a10), 2);
            }
            f16945l = bArr;
        }
        return f16945l;
    }

    private static Enumeration<String> c() {
        return Collections.enumeration(Arrays.asList(ATPCAMSConnectSetting.class.getName(), ATPMobileATP.class.getName(), ATPProxySetting.class.getName(), ATPResult.class.getName(), jp.co.canon.oip.android.opal.mobileatp.a.a.a.class.getName(), jp.co.canon.oip.android.opal.mobileatp.a.a.b.class.getName(), jp.co.canon.oip.android.opal.mobileatp.a.a.c.class.getName(), jp.co.canon.oip.android.opal.mobileatp.a.a.d.class.getName(), jp.co.canon.oip.android.opal.mobileatp.a.a.e.class.getName(), jp.co.canon.oip.android.opal.mobileatp.a.b.a.class.getName(), jp.co.canon.oip.android.opal.mobileatp.a.b.b.class.getName(), jp.co.canon.oip.android.opal.mobileatp.a.b.c.class.getName(), jp.co.canon.oip.android.opal.mobileatp.a.b.d.class.getName(), jp.co.canon.oip.android.opal.mobileatp.a.b.e.class.getName(), jp.co.canon.oip.android.opal.mobileatp.a.c.a.class.getName(), jp.co.canon.oip.android.opal.mobileatp.b.a.class.getName(), jp.co.canon.oip.android.opal.mobileatp.deviceregistration.a.class.getName(), ATPDeviceRegistrationRequest.class.getName(), ATPException.class.getName(), jp.co.canon.oip.android.opal.mobileatp.c.a.class.getName(), jp.co.canon.oip.android.opal.mobileatp.c.b.class.getName(), jp.co.canon.oip.android.opal.mobileatp.c.c.class.getName(), jp.co.canon.oip.android.opal.mobileatp.c.d.class.getName(), jp.co.canon.oip.android.opal.mobileatp.d.a.class.getName(), jp.co.canon.oip.android.opal.mobileatp.d.b.class.getName(), jp.co.canon.oip.android.opal.mobileatp.e.a.class.getName(), jp.co.canon.oip.android.opal.mobileatp.f.a.class.getName(), jp.co.canon.oip.android.opal.mobileatp.f.b.class.getName(), jp.co.canon.oip.android.opal.mobileatp.token.a.class.getName(), ATPResultAccessToken.class.getName(), b.class.getName(), c.class.getName(), d.class.getName(), e.class.getName(), f.class.getName(), g.class.getName(), jp.co.canon.oip.android.opal.mobileatp.util.a.class.getName()));
    }

    private static byte[] a(byte[] bArr, byte[] bArr2, int i9) {
        try {
            SecretKeySpec secretKeySpec = new SecretKeySpec(bArr, f16941h);
            try {
                Cipher cipher = Cipher.getInstance(f16942i);
                cipher.init(i9, secretKeySpec, new IvParameterSpec(g.a(f16940g)));
                return cipher.doFinal(bArr2);
            } catch (IllegalStateException e10) {
                if (i9 == 1) {
                    throw e10;
                }
                throw new ATPException(ATPResult.RESULT_CODE_NG_DECRYPT_KEY, "invalid decrypt key");
            }
        } catch (InvalidAlgorithmParameterException e11) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_CRYPTO_INVALID_ALGORITHM_PARAM, e11.getMessage(), e11);
        } catch (InvalidKeyException e12) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_CRYPTO_INVALID_KEY, e12.getMessage(), e12);
        } catch (NoSuchAlgorithmException e13) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_CRYPTO_NO_SUCH_ALGORITHM, e13.getMessage(), e13);
        } catch (BadPaddingException e14) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_CRYPTO_BAD_PADDING, e14.getMessage(), e14);
        } catch (IllegalBlockSizeException e15) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_CRYPTO_BLOCK_SIZE, e15.getMessage(), e15);
        } catch (NoSuchPaddingException e16) {
            throw new ATPException(404, e16.getMessage(), e16);
        }
    }

    public static String[] b(int i9) {
        if (i9 == 0) {
            return new String[]{jp.co.canon.oip.android.opal.mobileatp.util.a.class.getName()};
        }
        if (i9 != 1) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(1, "default p[" + i9 + ']');
            return null;
        }
        return new String[]{jp.co.canon.oip.android.opal.mobileatp.util.a.class.getName(), ArrayList.class.getName()};
    }

    public static void a() {
        f16945l = null;
    }

    public static String a(int i9) {
        int i10;
        String a10;
        try {
            HashMap hashMap = new HashMap();
            Enumeration<String> c10 = c();
            String name = c.class.getPackage().getName();
            String substring = name.substring(0, name.lastIndexOf(46));
            while (c10.hasMoreElements()) {
                String nextElement = c10.nextElement();
                if (nextElement.startsWith(substring) && (a10 = a(nextElement, i9)) != null && a10.length() > 2) {
                    int parseInt = Integer.parseInt(a10.substring(a10.length() - 2), 16);
                    if (hashMap.get(Integer.valueOf(parseInt)) == null) {
                        hashMap.put(Integer.valueOf(parseInt), a10.substring(0, a10.length() - 2));
                    } else {
                        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "n:" + parseInt + " duplicate");
                    }
                }
            }
            StringBuilder sb = new StringBuilder();
            int i11 = 0;
            int i12 = 0;
            for (int i13 = 0; i13 < 256; i13++) {
                String str = (String) hashMap.get(Integer.valueOf(i13));
                if (str != null) {
                    if (i11 == 0) {
                        i11 = Integer.parseInt(str.substring(0, 2), 16);
                        i10 = 2;
                    } else {
                        i10 = 0;
                    }
                    if (i10 < str.length()) {
                        int i14 = i11 - i12;
                        String substring2 = str.substring(i10, str.length());
                        if (substring2.length() > i14) {
                            substring2 = str.substring(i10, i14 + i10);
                        }
                        sb.append(substring2);
                        i12 += substring2.length();
                    }
                    if (i12 >= i11) {
                        break;
                    }
                }
            }
            if (i11 != 0 && i11 == i12) {
                return sb.toString();
            }
            throw new ATPException(ATPResult.RESULT_CODE_NG_CRYPTO_INVALID_KEY, "Crypto Key size error.");
        } catch (Exception unused) {
            return null;
        }
    }

    public static String c(int i9) {
        if (i9 != 0 && i9 != 1) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(1, "default r[" + i9 + ']');
            return null;
        }
        return String.class.getName();
    }

    private static String a(String str, int i9) {
        try {
            Method[] methods = c.class.getClassLoader().loadClass(str).getMethods();
            String c10 = c(i9);
            String[] b10 = b(i9);
            int length = methods.length;
            if (c10 == null || b10 == null) {
                return null;
            }
            int length2 = b10.length;
            for (int i10 = 0; i10 < length; i10++) {
                Class<?>[] parameterTypes = methods[i10].getParameterTypes();
                Class<?> returnType = methods[i10].getReturnType();
                if (parameterTypes != null) {
                    int length3 = parameterTypes.length;
                    if (returnType.getName().equals(c10) && length3 == length2) {
                        int i11 = 0;
                        while (i11 < length3 && parameterTypes[i11].getName().equals(b10[i11])) {
                            i11++;
                        }
                        if (i11 == length3) {
                            Object[] objArr = new Object[length3];
                            for (int i12 = 0; i12 < length3; i12++) {
                                objArr[i12] = parameterTypes[i12].newInstance();
                            }
                            return (String) methods[i10].invoke(null, objArr);
                        }
                    }
                }
            }
            return null;
        } catch (ClassNotFoundException e10) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(e10);
            return null;
        } catch (IllegalAccessException e11) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(e11);
            return null;
        } catch (InstantiationException e12) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(e12);
            return null;
        } catch (OutOfMemoryError e13) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(e13);
            return null;
        } catch (InvocationTargetException e14) {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(e14);
            return null;
        }
    }
}
