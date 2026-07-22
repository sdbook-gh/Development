package jp.co.canon.bsd.ad.sdk.core.network;

import java.io.BufferedInputStream;
import java.io.OutputStream;
import java.security.SecureRandom;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

/* loaded from: /mnt/f/print/classes3.dex */
public class SslSocket {
    private SSLSocket mSslSocket;
    private static List<SslSocket> instanceList = new ArrayList();
    private static int SSL_TIMEOUT = 10000;
    private static int SSL_PORT = 443;
    private static String HTTPS_HEADER = "https://";
    private String mSocketId = UUID.randomUUID().toString();
    private OutputStream mOutputStream = null;
    private BufferedInputStream mBufferedIntputStream = null;
    private String mConnectIpAddr = "";

    public class a implements X509TrustManager {
        @Override
        public final void checkClientTrusted(X509Certificate[] chain, String authType)
                throws CertificateException {
            // Trust-all for Canon LAN SSL helper (needed for JNI class load).
        }

        @Override
        public final void checkServerTrusted(X509Certificate[] chain, String authType)
                throws CertificateException {
            // Trust-all for Canon LAN SSL helper (needed for JNI class load).
        }

        @Override
        public final X509Certificate[] getAcceptedIssuers() {
            return new X509Certificate[0];
        }
    }

    private SslSocket() {
    }

    private void closeSocket() {
        OutputStream outputStream = this.mOutputStream;
        if (outputStream != null) {
            try {
                outputStream.close();
            } catch (Exception unused) {
            }
            this.mOutputStream = null;
        }
        BufferedInputStream bufferedInputStream = this.mBufferedIntputStream;
        if (bufferedInputStream != null) {
            try {
                bufferedInputStream.close();
            } catch (Exception unused2) {
            }
            this.mBufferedIntputStream = null;
        }
        SSLSocket sSLSocket = this.mSslSocket;
        if (sSLSocket != null) {
            try {
                sSLSocket.close();
            } catch (Exception unused3) {
            }
            this.mSslSocket = null;
        }
        this.mConnectIpAddr = "";
        for (int i9 = 0; i9 < instanceList.size(); i9++) {
            if (this.mSocketId.equals(instanceList.get(i9).mSocketId)) {
                instanceList.remove(i9);
                return;
            }
        }
    }

    public static SslSocket getInstance() {
        SslSocket sslSocket = new SslSocket();
        instanceList.add(sslSocket);
        return sslSocket;
    }

    public int connectSsl(String str) {
        int i9 = 0;
        try {
            TrustManager[] trustManagerArr = {new a()};
            SSLContext sSLContext = SSLContext.getInstance("SSL");
            sSLContext.init(null, trustManagerArr, new SecureRandom());
            SSLSocket sSLSocket = (SSLSocket) sSLContext.getSocketFactory().createSocket(str, SSL_PORT);
            this.mSslSocket = sSLSocket;
            if (sSLSocket != null) {
                sSLSocket.setSoTimeout(SSL_TIMEOUT);
                this.mConnectIpAddr = str;
            } else {
                i9 = -1;
            }
            return i9;
        } catch (Exception unused) {
            return -1;
        }
    }

    public byte[] recvData(int i9) {
        byte[] bArr = new byte[1];
        try {
            if (this.mBufferedIntputStream == null) {
                if (this.mSslSocket == null) {
                    return null;
                }
                this.mBufferedIntputStream = new BufferedInputStream(this.mSslSocket.getInputStream());
            }
            int i10 = 0;
            do {
                int read = this.mBufferedIntputStream.read(bArr, i10, 1 - i10);
                if (read == -1) {
                    break;
                }
                i10 += read;
            } while (i10 < 1);
            if (bArr[0] == 0) {
                return null;
            }
            return bArr;
        } catch (Exception unused) {
            return null;
        }
    }

    public int sendData(String str) {
        try {
            if (this.mOutputStream == null) {
                this.mOutputStream = this.mSslSocket.getOutputStream();
            }
            this.mOutputStream.write(str.replace(HTTPS_HEADER + this.mConnectIpAddr, "").getBytes("UTF-8"));
            this.mOutputStream.flush();
        } catch (Exception e10) {
            e10.printStackTrace();
        }
        return str.length();
    }
}
