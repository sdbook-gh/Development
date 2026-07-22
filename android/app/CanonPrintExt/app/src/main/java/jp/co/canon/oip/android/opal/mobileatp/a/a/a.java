package jp.co.canon.oip.android.opal.mobileatp.a.a;

import android.net.Uri;
import b.C1204a;
import b.C1205b;
import com.google.firebase.messaging.Constants;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.InterruptedIOException;
import java.net.MalformedURLException;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.net.UnknownHostException;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.Properties;
import javax.net.ssl.SSLHandshakeException;
import javax.net.ssl.SSLPeerUnverifiedException;
import jp.co.canon.android.cnml.common.CNMLJCmnUtil;
import jp.co.canon.oip.android.opal.mobileatp.ATPCAMSConnectSetting;
import jp.co.canon.oip.android.opal.mobileatp.ATPProxySetting;
import jp.co.canon.oip.android.opal.mobileatp.ATPResult;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import jp.co.canon.oip.android.opal.mobileatp.util.f;
import jp.co.canon.oip.android.opal.mobileatp.util.g;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: ATPAbstractCAMSProcess.java */
public abstract class a {

    private static long f16827c;
    private static long f16828d;

    protected ATPProxySetting f16829a = null;
    protected ATPCAMSConnectSetting f16830b;

    public class C0318a {
        public final boolean f16831a;
        public final ATPException f16832b;
        public final jp.co.canon.oip.android.opal.mobileatp.a.b.d f16833c;

        public C0318a(int i9, String str) {
            boolean ok = i9 == 200;
            ATPException error = null;
            jp.co.canon.oip.android.opal.mobileatp.a.b.d parsed = null;
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "start");
            JSONObject jSONObject = null;
            try {
                jSONObject = new JSONObject(str);
            } catch (JSONException unused) {
            }
            if (ok) {
                try {
                    parsed = a.this.a(jSONObject);
                } catch (ATPException e10) {
                    error = e10;
                    ok = false;
                } catch (Exception e11) {
                    error = new ATPException(902, "Analyse failed.", e11);
                    ok = false;
                }
            } else {
                jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, str);
                if (jSONObject != null) {
                    try {
                        String code;
                        String message;
                        try {
                            code = jSONObject.getString("code");
                            message = jSONObject.getString("message");
                        } catch (JSONException unused2) {
                            code = jSONObject.getString(Constants.IPC_BUNDLE_KEY_SEND_ERROR);
                            message = jSONObject.optString("error_description", "");
                        }
                        if (g.a(code)) {
                            error = new ATPException(2, i9, "can't get json error.");
                        } else {
                            error = new ATPException(1, i9, code, g.a(message) ? "" : message);
                        }
                    } catch (Exception e13) {
                        error = new ATPException(902, "Analyse failed.", e13);
                    }
                } else {
                    error = new ATPException(2, i9, "not kind of json error.");
                }
            }
            this.f16831a = ok && error == null;
            this.f16832b = error;
            this.f16833c = parsed;
        }
    }

    public a() {
        this.f16830b = jp.co.canon.oip.android.opal.mobileatp.f.b.g().e();
    }

    private C1204a h() {
        C1204a c1204a = new C1204a();
        HashMap<String, String> hashMap = new HashMap<>();
        c1204a.f9775n = hashMap;
        c1204a.f9762a = this.f16830b.getConTimeout();
        c1204a.f9763b = this.f16830b.getSoTimeout();
        c1204a.f9767f = g();
        c1204a.f9768g = c();
        if (a(this.f16829a)) {
            c1204a.f9769h = this.f16829a.getHost();
            c1204a.f9770i = this.f16829a.getPort();
            c1204a.f9773l = this.f16829a.getUser();
            c1204a.f9774m = this.f16829a.getPassword();
        }
        String d10 = d();
        if (d10 != null) {
            byte[] bytes = d10.getBytes(Charset.forName("UTF-8"));
            if (!hashMap.containsKey(e.f16855l)) {
                hashMap.put(e.f16855l, Integer.toString(bytes.length));
            }
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            c1204a.f9764c = byteArrayOutputStream;
            try {
                byteArrayOutputStream.write(bytes);
                byteArrayOutputStream.flush();
            } catch (IOException e10) {
                throw new ATPException(ATPResult.RESULT_CODE_NG_REQUEST_IO_EXCEPTION, e10);
            }
            c1204a.f9765d = new ByteArrayInputStream(byteArrayOutputStream.toByteArray());
        }
        c1204a.f9766e = "POST";
        String userAgent = this.f16830b.getUserAgent();
        if (userAgent == null || userAgent.isEmpty()) {
            throw new ATPException(107);
        }
        c1204a.d("User-Agent", userAgent);
        Properties e10 = e();
        if (e10 != null) {
            for (String str : e10.stringPropertyNames()) {
                String property = e10.getProperty(str, null);
                if (!e.f16855l.equals(str) && property != null) {
                    c1204a.d(str, property);
                }
            }
        }
        return c1204a;
    }

    private void i() {
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "HttpRequest total time(msec): ", f16828d - f16827c);
    }

    private void l() {
        f16828d = System.currentTimeMillis();
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "end time(msec): " + f16828d);
    }

    private void m() {
        f16827c = System.currentTimeMillis();
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "start time(msec): " + f16827c);
    }

    public final URL a(String str, String str2) {
        try {
            StringBuilder c10 = C3.d.c(str);
            c10.append(f());
            Uri.Builder buildUpon = Uri.parse(c10.toString()).buildUpon();
            if (str2 != null && !str2.isEmpty()) {
                buildUpon.appendQueryParameter("realm", str2);
            }
            URL url = new URL(buildUpon.build().toString());
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, str);
            return url;
        } catch (MalformedURLException e10) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_CREATE_REQUEST_MESSAGE, e10);
        }
    }

    public abstract jp.co.canon.oip.android.opal.mobileatp.a.b.d a(JSONObject jSONObject);

    public abstract URL b();

    public String c() {
        return null;
    }

    public String d() {
        return null;
    }

    public Properties e() {
        return null;
    }

    public abstract String f();

    public String g() {
        return null;
    }

    public jp.co.canon.oip.android.opal.mobileatp.a.b.d j() {
        try {
            C0318a k9 = k();
            if (k9 == null) {
                throw new ATPException(102, "cams access.");
            }
            if (!k9.f16831a) {
                if (k9.f16832b != null) {
                    throw k9.f16832b;
                }
                throw new ATPException(102, "cams access.");
            }
            jp.co.canon.oip.android.opal.mobileatp.a.b.d dVar = k9.f16833c;
            if (dVar != null) {
                return dVar;
            }
            throw new ATPException(102, "cams response object.");
        } catch (ATPException e10) {
            throw e10;
        } catch (Exception e11) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_CAMS_ACCESS, e11.getMessage(), e11);
        }
    }

    public final C0318a k() {
        try {
            return a();
        } catch (SocketTimeoutException e11) {
            e11.printStackTrace();
            throw new ATPException(ATPResult.RESULT_CODE_NG_REQUEST_CONNECT_TIMEOUT, e11);
        } catch (IOException e12) {
            throw new ATPException(ATPResult.RESULT_CODE_NG_REQUEST_IO_EXCEPTION, e12);
        }
    }

    private String a(InputStream inputStream, String str) throws IOException {
        Charset charset;
        try {
            charset = Charset.forName(str);
        } catch (Exception unused) {
            charset = Charset.forName("UTF-8");
        }
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream, charset));
        StringBuilder sb = new StringBuilder();
        try {
            String readLine;
            while ((readLine = bufferedReader.readLine()) != null) {
                sb.append(readLine).append(CNMLJCmnUtil.LF);
            }
        } finally {
            try {
                bufferedReader.close();
            } catch (IOException unused2) {
            }
            try {
                inputStream.close();
            } catch (IOException unused3) {
            }
        }
        return sb.toString();
    }

    private synchronized void a(long j9) {
        try {
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "start.");
            wait(j9);
            jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "end.");
        } catch (InterruptedException unused) {
        }
    }

    private C0318a a() throws IOException {
        int retryCount = this.f16830b.getRetryCount();
        long retryInterval = this.f16830b.getRetryInterval();
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "retryCount: " + retryCount + ", retryInterval: " + retryInterval);
        URL b10 = b();
        C0318a last = null;
        for (int i10 = 0; i10 < retryCount; i10++) {
            C1204a c1204a = null;
            try {
                m();
                c1204a = h();
                C1205b a10 = c1204a.a(b10);
                int i9 = a10.f9781a;
                String str2 = a10.f9784d;
                // Always read body: CAMS error JSON lives on non-200; discarding it
                // produced opaque "not kind of json error" on 401.
                String str = a(a10.f9783c, C1205b.a(str2));
                last = new C0318a(i9, str);
                l();
                i();
                if (!f.a(i9) || i10 >= retryCount - 1) {
                    return last;
                }
                a(retryInterval);
            } catch (IOException e10) {
                l();
                i();
                if (!a(e10, retryCount - 1, i10)) {
                    throw e10;
                }
                a(retryInterval);
            } finally {
                if (c1204a != null) {
                    c1204a.e();
                }
            }
        }
        return last;
    }

    private boolean a(IOException iOException, int i9, int i10) {
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "retryCount: " + i9 + ", executionCount: " + i10);
        jp.co.canon.oip.android.opal.mobileatp.d.b.a(3, "exception :" + iOException.getMessage());
        if (i10 >= i9) {
            return false;
        }
        if (iOException instanceof SocketTimeoutException) {
            return true;
        }
        if ((iOException instanceof InterruptedIOException)
                || (iOException instanceof UnknownHostException)
                || (iOException instanceof SSLHandshakeException)
                || (iOException instanceof SSLPeerUnverifiedException)) {
            return false;
        }
        return true;
    }

    private boolean a(ATPProxySetting aTPProxySetting) {
        return (aTPProxySetting == null || aTPProxySetting.getHost() == null) ? false : true;
    }
}
