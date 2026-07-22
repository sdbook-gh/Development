package jp.co.canon.oip.android.opal.mobileatp.d;

import java.io.PrintWriter;
import java.io.StringWriter;

/* compiled from: ATPLogger.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class b {

    /* renamed from: a, reason: collision with root package name */
    private static final String f16900a = "ATPMobileATP";

    /* renamed from: b, reason: collision with root package name */
    private static final int f16901b = 1;

    /* renamed from: c, reason: collision with root package name */
    private static final int f16902c = 2;

    /* renamed from: d, reason: collision with root package name */
    private static final String f16903d = ":";

    /* renamed from: e, reason: collision with root package name */
    private static final String f16904e = ".";

    /* renamed from: f, reason: collision with root package name */
    private static final String[] f16905f = {"SYS", "ERR", "INF", "DBG"};

    /* renamed from: g, reason: collision with root package name */
    private static final Object f16906g = new Object();

    /* renamed from: h, reason: collision with root package name */
    private static final boolean[] f16907h = {true, false, false, false, false, false, false, false};

    public static void a() {
    }

    public static int b() {
        return f16905f.length - 1;
    }

    private static boolean a(int i9) {
        return i9 >= 0 && i9 <= 3 && i9 <= -1;
    }

    public static synchronized void a(int i9, int i10) {
        synchronized (b.class) {
            if (a(i9)) {
                synchronized (f16906g) {
                    StackTraceElement stackTraceElement = new Throwable().getStackTrace()[1];
                    a(0, i9, String.valueOf(i10), a(stackTraceElement.getClassName()), stackTraceElement.getMethodName());
                }
            }
        }
    }

    public static synchronized void a(int i9, String str) {
        synchronized (b.class) {
            if (a(i9)) {
                synchronized (f16906g) {
                    StackTraceElement stackTraceElement = new Throwable().getStackTrace()[1];
                    a(0, i9, str, a(stackTraceElement.getClassName()), stackTraceElement.getMethodName());
                }
            }
        }
    }

    public static synchronized void a(int i9, String str, long j9) {
        synchronized (b.class) {
            if (a(i9)) {
                String str2 = str + Long.toString(j9);
                synchronized (f16906g) {
                    StackTraceElement stackTraceElement = new Throwable().getStackTrace()[1];
                    a(0, i9, str2, a(stackTraceElement.getClassName()), stackTraceElement.getMethodName());
                }
            }
        }
    }

    public static synchronized void a(Throwable th) {
        synchronized (b.class) {
            if (a(1)) {
                synchronized (f16906g) {
                    try {
                        StackTraceElement stackTraceElement = new Throwable().getStackTrace()[1];
                        String methodName = stackTraceElement.getMethodName();
                        String a10 = a(stackTraceElement.getClassName());
                        StringBuilder sb = new StringBuilder();
                        if (th != null) {
                            sb.append(a(th.getClass().getName()));
                            sb.append(" \"");
                            sb.append(th.getMessage());
                            sb.append("\"\n");
                            StringWriter stringWriter = new StringWriter();
                            PrintWriter printWriter = new PrintWriter(stringWriter);
                            th.printStackTrace(printWriter);
                            printWriter.flush();
                            String stringWriter2 = stringWriter.toString();
                            printWriter.close();
                            sb.append(stringWriter2);
                        } else {
                            sb.append("null");
                        }
                        a(0, 1, sb.toString(), a10, methodName);
                    } finally {
                    }
                }
            }
        }
    }

    public static void c() {
    }

    public static synchronized void a(int i9, Throwable th) {
        synchronized (b.class) {
            if (a(1)) {
                synchronized (f16906g) {
                    try {
                        StackTraceElement stackTraceElement = new Throwable().getStackTrace()[2];
                        String methodName = stackTraceElement.getMethodName();
                        String a10 = a(stackTraceElement.getClassName());
                        StringBuilder sb = new StringBuilder("[ID=");
                        sb.append(i9);
                        sb.append(']');
                        if (th != null) {
                            sb.append(a(th.getClass().getName()));
                            sb.append(" \"");
                            sb.append(th.getMessage());
                            sb.append("\"\n");
                            StringWriter stringWriter = new StringWriter();
                            PrintWriter printWriter = new PrintWriter(stringWriter);
                            th.printStackTrace(printWriter);
                            printWriter.flush();
                            String stringWriter2 = stringWriter.toString();
                            printWriter.close();
                            sb.append(stringWriter2);
                        }
                        a(0, 1, sb.toString(), a10, methodName);
                    } catch (Throwable th2) {
                        throw th2;
                    }
                }
            }
        }
    }

    public static synchronized void a(int i9, Throwable th, Throwable th2) {
        synchronized (b.class) {
            if (a(1)) {
                synchronized (f16906g) {
                    try {
                        StackTraceElement stackTraceElement = new Throwable().getStackTrace()[2];
                        String methodName = stackTraceElement.getMethodName();
                        String a10 = a(stackTraceElement.getClassName());
                        StringBuilder sb = new StringBuilder("[ID=");
                        sb.append(i9);
                        sb.append(']');
                        if (th != null) {
                            sb.append(a(th.getClass().getName()));
                            sb.append(" \"");
                            sb.append(th.getMessage());
                            sb.append('\"');
                        }
                        if (th2 != null) {
                            sb.append('[');
                            sb.append(a(th2.getClass().getName()));
                            sb.append(" \"");
                            sb.append(th2.getMessage());
                            sb.append("\"]");
                        }
                        if (th != null) {
                            StringWriter stringWriter = new StringWriter();
                            PrintWriter printWriter = new PrintWriter(stringWriter);
                            th.printStackTrace(printWriter);
                            printWriter.flush();
                            String stringWriter2 = stringWriter.toString();
                            printWriter.close();
                            sb.append('\n');
                            sb.append(stringWriter2);
                        }
                        a(0, 1, sb.toString(), a10, methodName);
                    } catch (Throwable th3) {
                        throw th3;
                    }
                }
            }
        }
    }

    private static String a(String str) {
        if (str == null) {
            return null;
        }
        return str.substring(str.lastIndexOf(".") + 1);
    }

    private static void a(int i9, int i10, String str, String str2, String str3) {
        if (i9 >= 0) {
            boolean[] zArr = f16907h;
            if (i9 >= zArr.length || str == null || !zArr[i9]) {
                return;
            }
            Thread.currentThread().getName();
            String str4 = f16905f[i10];
        }
    }
}
