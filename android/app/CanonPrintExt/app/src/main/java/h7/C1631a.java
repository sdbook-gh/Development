package h7;

import android.util.Log;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketTimeoutException;
import jp.co.canon.android.cnml.device.type.CNMLDeviceStatusCodeType;
import jp.co.canon.android.cnml.file.type.CNMLFileType;
import k7.h;

/* compiled from: BjnpSocket.java */
/* renamed from: h7.a, reason: case insensitive filesystem */
/* loaded from: /mnt/f/print/classes3.dex */
public final class C1631a implements InterfaceC1632b {

    private static final String TAG = "G3800Bjnp";

    /** Decompiler mapped this to HI_BYTE_MASK (= 0xFF00). */
    private static final int HI_BYTE_MASK = 0xFF00;
    /** Decompiler mapped this to CONNECT_TIMEOUT_MS (= PathInterpolatorCompat.MAX_NUM_POINTS = 3000). */
    private static final int CONNECT_TIMEOUT_MS = 3000;
    /**
     * Official BjnpSocket session-open UDP length
     * ({@code ATPResult.RESULT_CODE_NG_CRYPTO_BAD_PADDING} = 408 = 16 hdr + 0x188 payload).
     * Earlier rewrite wrongly sent only 24 bytes and truncated host/user/document identity.
     */
    private static final int SESSION_OPEN_UDP_LEN = 408;

    /* renamed from: a, reason: collision with root package name */
    public final int f14226a;

    /* renamed from: b, reason: collision with root package name */
    public final byte[] f14227b;

    /* renamed from: c, reason: collision with root package name */
    public final byte[] f14228c;

    /* renamed from: d, reason: collision with root package name */
    public int f14229d;

    /* renamed from: e, reason: collision with root package name */
    public Socket f14230e;

    /* renamed from: f, reason: collision with root package name */
    public DataInputStream f14231f;

    /* renamed from: g, reason: collision with root package name */
    public DataOutputStream f14232g;

    /* renamed from: h, reason: collision with root package name */
    public int f14233h;

    /* renamed from: i, reason: collision with root package name */
    public String f14234i;

    /* renamed from: j, reason: collision with root package name */
    public int f14235j;

    /* renamed from: k, reason: collision with root package name */
    public boolean f14236k;

    /* renamed from: l, reason: collision with root package name */
    public long f14237l;

    /* renamed from: m, reason: collision with root package name */
    public long f14238m;

    /* renamed from: n, reason: collision with root package name */
    public static final byte[] f14213n = {0, 97, 0, 110, 0, 100, 0, 114, 0, 111, 0, 105, 0, 100, 0, 32};

    /* renamed from: o, reason: collision with root package name */
    public static final byte[] f14214o = {0, 97, 0, 110, 0, 100, 0, 114, 0, 111, 0, 105, 0, 100, 0, 32, 0, 117, 0, 115, 0, 101, 0, 114};

    /* renamed from: p, reason: collision with root package name */
    public static final byte[] f14215p = {0, 97, 0, 110, 0, 100, 0, 114, 0, 111, 0, 105, 0, 100, 0, 32, 0, 100, 0, 111, 0, 99, 0, 117, 0, 109, 0, 101, 0, 110, 0, 116};

    /* renamed from: q, reason: collision with root package name */
    public static final byte[] f14216q = {0, 97, 0, 110, 0, 100, 0, 114, 0, 111, 0, 105, 0, 100, 0, 32, 0, 115, 0, 99, 0, 97, 0, 110, 0, 110, 0, 101, 0, 114, 0, 0};

    /* renamed from: r, reason: collision with root package name */
    public static final byte[] f14217r = {66, 74, 78, 80, 1, 16, 0, 0, 0, 0, 0, 0, 0, 0, 1, -120, 0, 0, 0, 0, 0, 0, 0, 0};

    /* renamed from: s, reason: collision with root package name */
    public static final byte[] f14218s = {66, 74, 78, 80, 1, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    /* renamed from: t, reason: collision with root package name */
    public static final byte[] f14219t = {66, 74, 78, 80, 1, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    /* renamed from: u, reason: collision with root package name */
    public static final byte[] f14220u = {66, 74, 78, 80, -127, 33, 0, 0};

    /* renamed from: v, reason: collision with root package name */
    public static final byte[] f14221v = {66, 74, 78, 80, -126, 33, 0, 0};

    /* renamed from: w, reason: collision with root package name */
    public static final byte[] f14222w = {66, 74, 78, 80, 1, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    /* renamed from: x, reason: collision with root package name */
    public static final byte[] f14223x = {66, 74, 78, 80};

    /* renamed from: y, reason: collision with root package name */
    public static final byte[] f14224y = {66, 74, 78, 80, -127, 32, 0, 0};

    /* renamed from: z, reason: collision with root package name */
    public static final byte[] f14225z = {66, 74, 78, 80, -126, 32, 0, 0};

    /* renamed from: A, reason: collision with root package name */
    public static final byte[] f14208A = {66, 74, 78, 80, 1, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    /* renamed from: B, reason: collision with root package name */
    public static final byte[] f14209B = {66, 74, 78, 80, -127, 49, 0, 0};

    /* renamed from: C, reason: collision with root package name */
    public static final byte[] f14210C = {66, 74, 78, 80, 2, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 30};

    /* renamed from: D, reason: collision with root package name */
    public static final byte[] f14211D = {66, 74, 78, 80, 1, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 30};

    /* renamed from: E, reason: collision with root package name */
    public static final byte[] f14212E = {66, 74, 78, 80, -126, 20, 0, 0};

    public C1631a() {
        this.f14229d = 1;
        this.f14236k = false;
        this.f14237l = 0L;
        this.f14238m = 0L;
        this.f14226a = 1;
        this.f14227b = new byte[24768];
        this.f14228c = new byte[24768];
        this.f14230e = null;
        this.f14231f = null;
        this.f14232g = null;
        this.f14233h = 0;
        this.f14234i = null;
        this.f14235j = 8611;
    }

    public final byte[] a(DataInputStream dataInputStream) {
        byte[] bArr = new byte[2048];
        byte[] bArr2 = null;
        while (!this.f14236k) {
            int read;
            try {
                read = dataInputStream.read(bArr);
            } catch (SocketTimeoutException e10) {
                e10.toString();
                h.p(500);
                continue;
            } catch (IOException e11) {
                e11.toString();
                return null;
            }
            if (read < 0 || (bArr2 = h.b(bArr2, bArr, read)) == null) {
                return null;
            }
            if (bArr2.length >= 16) {
                byte[] e12 = h.e(f14223x, bArr2, 4);
                if (e12 == null) {
                    return null;
                }
                int i9 = ((e12[14] & 255) * 256) + ((e12[13] & 255) * 65536) + ((e12[12] & 255) * 16777216) + (e12[15] & 255);
                while (e12.length < i9 + 16) {
                    if (this.f14236k) {
                        return null;
                    }
                    try {
                        int n = dataInputStream.read(bArr);
                        e12 = h.b(e12, bArr, n);
                    } catch (SocketTimeoutException e13) {
                        e13.toString();
                        h.p(500);
                        continue;
                    } catch (IOException e14) {
                        e14.toString();
                        return null;
                    }
                    if (e12 == null) {
                        return null;
                    }
                }
                return e12;
            }
            h.p(500);
        }
        return null;
    }

    public final void b(boolean z9) {
        String str;
        int i9;
        DatagramSocket datagramSocket;
        try {
            if (this.f14230e != null) {
                if (z9) {
                    h.p(40000);
                }
                this.f14230e.close();
            }
        } catch (IOException unused) {
        }
        int i10 = this.f14233h;
        if (i10 != 0 && (str = this.f14234i) != null && (i9 = this.f14235j) != 0 && !z9) {
            System.arraycopy(f14218s, 0, this.f14228c, 0, 16);
            int c10 = c();
            byte[] bArr = this.f14228c;
            bArr[8] = (byte) ((c10 & HI_BYTE_MASK) >>> 8);
            bArr[9] = (byte) (c10 & 255);
            bArr[10] = (byte) ((65280 & i10) >>> 8);
            bArr[11] = (byte) (i10 & 255);
            boolean z10 = true;
            if (this.f14226a == 2) {
                bArr[4] = 2;
                bArr[5] = 17;
            } else {
                bArr[4] = 1;
                bArr[5] = 17;
            }
            try {
                datagramSocket = new DatagramSocket();
                try {
                    datagramSocket.setSoTimeout(1000);
                    datagramSocket.send(new DatagramPacket(this.f14228c, 16, InetAddress.getByName(str), i9));
                    h.o();
                    byte[] bArr2 = this.f14227b;
                    DatagramPacket datagramPacket = new DatagramPacket(bArr2, bArr2.length);
                    int i11 = 0;
                    while (true) {
                        if (i11 >= 10) {
                            z10 = false;
                            break;
                        }
                        try {
                            datagramSocket.receive(datagramPacket);
                            break;
                        } catch (IOException e10) {
                            e10.toString();
                            h.o();
                            i11++;
                        }
                    }
                    datagramSocket.close();
                    if (z10) {
                        byte[] bArr3 = this.f14227b;
                        int i12 = ((bArr3[6] & 255) * 256) + (bArr3[7] & 255);
                        if (((bArr3[8] & 255) * 256) + (bArr3[9] & 255) == c10 && i12 == 0) {
                            byte b10 = bArr3[10];
                            byte b11 = bArr3[11];
                        }
                    }
                } catch (IOException unused2) {
                    if (datagramSocket != null) {
                        datagramSocket.close();
                    }
                    this.f14230e = null;
                    this.f14231f = null;
                    this.f14232g = null;
                    this.f14233h = 0;
                    this.f14234i = null;
                    this.f14235j = 0;
                }
            } catch (IOException unused3) {
                datagramSocket = null;
            }
        }
        this.f14230e = null;
        this.f14231f = null;
        this.f14232g = null;
        this.f14233h = 0;
        this.f14234i = null;
        this.f14235j = 0;
    }

    public final int c() {
        int i9 = this.f14229d + 1;
        this.f14229d = i9;
        if (1 > i9 || 65535 < i9) {
            this.f14229d = 1;
        }
        return this.f14229d;
    }

    @Override // h7.InterfaceC1632b
    public final void close() {
        b(false);
    }

    /**
     * BJNP session open: UDP handshake (408 B) → TCP connect → idle configure.
     * Aligned with Canon BjnpSocket; rewritten from broken jadx output.
     */
    public final int d(int port, String ip) {
        if (ip == null || this.f14230e != null || this.f14231f != null || this.f14232g != null) {
            Log.e(TAG, "[open] refuse rc=-2 ip=" + ip
                    + " sock=" + (this.f14230e != null)
                    + " in=" + (this.f14231f != null)
                    + " out=" + (this.f14232g != null));
            return -2;
        }
        this.f14234i = ip;
        this.f14235j = port;
        // Clear identity region so prior garbage is not sent in the 408-byte datagram.
        java.util.Arrays.fill(this.f14228c, 0, SESSION_OPEN_UDP_LEN, (byte) 0);
        System.arraycopy(f14217r, 0, this.f14228c, 0, 24);
        int seq = c();
        byte[] pkt = this.f14228c;
        pkt[8] = (byte) ((65280 & seq) >>> 8);
        pkt[9] = (byte) (seq & 255);
        int mode = this.f14226a;
        if (mode == 2) {
            pkt[4] = 2;
            pkt[5] = 16;
            System.arraycopy(f14213n, 0, pkt, 24, 16);
            System.arraycopy(f14214o, 0, this.f14228c, 88, 24);
            System.arraycopy(f14216q, 0, this.f14228c, 152, 32);
        } else {
            pkt[4] = 1;
            pkt[5] = 16;
            System.arraycopy(f14213n, 0, pkt, 24, 16);
            System.arraycopy(f14214o, 0, this.f14228c, 88, 24);
            System.arraycopy(f14215p, 0, this.f14228c, 152, 32);
        }
        if (this.f14237l > 0) {
            long retry = this.f14237l;
            pkt[16] = (byte) ((retry & CNMLFileType.MASK_PDF_MAC) >>> 24);
            pkt[17] = (byte) ((retry & CNMLDeviceStatusCodeType.NATIVE_MASK_FAX) >>> 16);
            pkt[18] = (byte) ((retry & CNMLDeviceStatusCodeType.NATIVE_MASK_PRINTER) >>> 8);
            pkt[19] = (byte) (retry & 255);
            long cookie = this.f14238m;
            pkt[20] = (byte) ((cookie & CNMLFileType.MASK_PDF_MAC) >>> 24);
            pkt[21] = (byte) ((cookie & CNMLDeviceStatusCodeType.NATIVE_MASK_FAX) >>> 16);
            pkt[22] = (byte) ((CNMLDeviceStatusCodeType.NATIVE_MASK_PRINTER & cookie) >>> 8);
            pkt[23] = (byte) (cookie & 255);
        }

        int payloadField = ((pkt[12] & 255) << 24)
                | ((pkt[13] & 255) << 16)
                | ((pkt[14] & 255) << 8)
                | (pkt[15] & 255);
        Log.i(TAG, "[open] UDP session-req → " + ip + ":" + port
                + " len=" + SESSION_OPEN_UDP_LEN
                + " mode=" + mode
                + " seq=" + seq
                + " payloadField=0x" + Integer.toHexString(payloadField)
                + " retry=" + this.f14237l
                + " cookie=" + this.f14238m
                + " head=" + hexPreview(pkt, 32));

        DatagramSocket udp = null;
        boolean gotReply = false;
        int recvLen = 0;
        String udpErr = null;
        long udpStart = System.currentTimeMillis();
        try {
            udp = new DatagramSocket();
            udp.setSoTimeout(1000);
            udp.send(new DatagramPacket(
                    this.f14228c,
                    SESSION_OPEN_UDP_LEN,
                    InetAddress.getByName(ip),
                    port));
            h.o();
            DatagramPacket recv = new DatagramPacket(this.f14227b, this.f14227b.length);
            for (int attempt = 0; attempt < 10; attempt++) {
                try {
                    udp.receive(recv);
                    gotReply = true;
                    recvLen = recv.getLength();
                    Log.i(TAG, "[open] UDP reply attempt=" + attempt
                            + " from=" + recv.getAddress()
                            + " len=" + recvLen
                            + " ms=" + (System.currentTimeMillis() - udpStart));
                    break;
                } catch (IOException e) {
                    Log.d(TAG, "[open] UDP recv wait attempt=" + attempt
                            + " " + e.getClass().getSimpleName() + ": " + e.getMessage());
                    h.o();
                }
            }
        } catch (IOException e) {
            gotReply = false;
            udpErr = e.getClass().getSimpleName() + ": " + e.getMessage();
            Log.e(TAG, "[open] UDP send/setup failed: " + udpErr, e);
        } finally {
            if (udp != null) {
                try {
                    udp.close();
                } catch (Exception ignored) {
                }
            }
        }

        if (!gotReply) {
            Log.e(TAG, "[open] no UDP reply from " + ip + ":" + port
                    + " after ~" + (System.currentTimeMillis() - udpStart) + "ms"
                    + (udpErr != null ? " err=" + udpErr : "")
                    + " → rc=-1");
            this.f14233h = 0;
            this.f14234i = null;
            this.f14235j = 0;
            return -1;
        }

        byte[] resp = this.f14227b;
        int payloadLen = ((resp[6] & 255) * 256) + (resp[7] & 255);
        int respSeq = ((resp[8] & 255) * 256) + (resp[9] & 255);
        int respSession = ((resp[10] & 255) * 256) + (resp[11] & 255);
        Log.i(TAG, "[open] UDP parse recvLen=" + recvLen
                + " payloadLen=" + payloadLen
                + " respSeq=" + respSeq + " (expect " + seq + ")"
                + " sessionHint=" + respSession
                + " head=" + hexPreview(resp, Math.min(32, recvLen)));

        if (respSeq != seq) {
            Log.e(TAG, "[open] seq mismatch → rc=-1");
            this.f14233h = 0;
            return -1;
        }
        if (payloadLen != 0) {
            // Busy / retry cookie from printer — caller should retry open on same instance.
            this.f14237l++;
            this.f14238m = ((resp[18] & 255) * 256L)
                    + ((resp[17] & 255) * 65536L)
                    + ((resp[16] & 255) * 16777216L)
                    + (resp[19] & 255);
            this.f14233h = 0;
            Log.w(TAG, "[open] printer busy payloadLen=" + payloadLen
                    + " nextRetry=" + this.f14237l
                    + " cookie=" + this.f14238m
                    + " → rc=-1 (retry)");
            return -1;
        }

        int sessionId = respSession;
        this.f14237l = 0L;
        this.f14238m = 0L;
        this.f14233h = sessionId;
        if (sessionId == 0) {
            Log.e(TAG, "[open] sessionId=0 → rc=-1");
            this.f14234i = null;
            this.f14235j = 0;
            return -1;
        }

        h.o();
        boolean ok = false;
        String tcpErr = null;
        long tcpStart = System.currentTimeMillis();
        try {
            Log.i(TAG, "[open] TCP connect → " + this.f14234i + ":" + this.f14235j
                    + " timeoutMs=" + CONNECT_TIMEOUT_MS
                    + " sessionId=" + sessionId);
            this.f14230e = new Socket();
            this.f14230e.connect(new InetSocketAddress(this.f14234i, this.f14235j), CONNECT_TIMEOUT_MS);
            if (mode == 1) {
                this.f14230e.setSoTimeout(1000);
            } else {
                this.f14230e.setSoTimeout(40000);
            }
            this.f14230e.setTcpNoDelay(true);
            this.f14231f = new DataInputStream(this.f14230e.getInputStream());
            this.f14232g = new DataOutputStream(this.f14230e.getOutputStream());
            h.o();
            int idle = mode == 1 ? 80 : 30;
            Log.i(TAG, "[open] TCP ok ms=" + (System.currentTimeMillis() - tcpStart)
                    + " idleConfigure=" + idle);
            if (mode == 1) {
                f(80);
            } else {
                f(30);
            }
            ok = true;
        } catch (IOException e) {
            ok = false;
            tcpErr = e.getClass().getSimpleName() + ": " + e.getMessage();
            Log.e(TAG, "[open] TCP/idle failed after "
                    + (System.currentTimeMillis() - tcpStart) + "ms: " + tcpErr, e);
        }
        if (!ok) {
            b(false);
            Log.e(TAG, "[open] → rc=-3 " + tcpErr);
            return -3;
        }
        Log.i(TAG, "[open] OK sessionId=" + sessionId + " " + ip + ":" + port);
        return 0;
    }

    private static String hexPreview(byte[] data, int max) {
        if (data == null || data.length == 0 || max <= 0) {
            return "(empty)";
        }
        int n = Math.min(data.length, max);
        StringBuilder sb = new StringBuilder(n * 3);
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                sb.append(' ');
            }
            sb.append(String.format("%02X", data[i] & 255));
        }
        if (data.length > max) {
            sb.append(" …(+").append(data.length - max).append(')');
        }
        return sb.toString();
    }

    public final boolean e() {
        if (this.f14230e == null || this.f14231f == null || this.f14232g == null || this.f14233h == 0) {
            return false;
        }
        c();
        try {
            System.arraycopy(f14208A, 0, this.f14228c, 0, 16);
            int c10 = c();
            byte[] bArr = this.f14228c;
            bArr[8] = (byte) ((c10 & HI_BYTE_MASK) >>> 8);
            bArr[9] = (byte) (c10 & 255);
            int i9 = this.f14233h;
            bArr[10] = (byte) ((65280 & i9) >>> 8);
            bArr[11] = (byte) (i9 & 255);
            this.f14232g.write(bArr, 0, 16);
            this.f14232g.flush();
            h.p(10);
            byte[] a10 = a(this.f14231f);
            if (a10 == null) {
                throw new IOException();
            }
            if (h.e(f14209B, a10, 8) != null) {
                return true;
            }
            throw new IOException();
        } catch (IOException unused) {
            return false;
        }
    }

    public final void f(int i9) {
        if (this.f14230e == null || this.f14231f == null || this.f14232g == null || this.f14233h == 0) {
            return;
        }
        try {
            if (this.f14226a == 2) {
                System.arraycopy(f14210C, 0, this.f14228c, 0, 20);
            } else {
                System.arraycopy(f14211D, 0, this.f14228c, 0, 20);
            }
            int c10 = c();
            byte[] bArr = this.f14228c;
            bArr[8] = (byte) ((c10 & HI_BYTE_MASK) >>> 8);
            bArr[9] = (byte) (c10 & 255);
            int i10 = this.f14233h;
            bArr[10] = (byte) ((65280 & i10) >>> 8);
            bArr[11] = (byte) (i10 & 255);
            bArr[16] = (byte) 0;
            bArr[17] = (byte) 0;
            bArr[18] = (byte) 0;
            bArr[19] = (byte) (i9 & 255);
            this.f14232g.write(bArr, 0, 20);
            this.f14232g.flush();
            h.p(10);
            byte[] a10 = a(this.f14231f);
            if (a10 == null) {
                throw new IOException();
            }
            if (h.e(f14212E, a10, 8) == null) {
                throw new IOException();
            }
        } catch (IOException unused) {
        }
    }

    @Override // h7.InterfaceC1632b
    public final int open(String str) {
        return d(8611, str);
    }

    @Override // h7.InterfaceC1632b
    public final byte[] read() {
        byte[] e10;
        if (this.f14230e == null || this.f14231f == null || this.f14232g == null || this.f14233h == 0) {
            return null;
        }
        try {
            System.arraycopy(f14222w, 0, this.f14228c, 0, 16);
            int c10 = c();
            byte[] bArr = this.f14228c;
            bArr[8] = (byte) ((c10 & HI_BYTE_MASK) >>> 8);
            bArr[9] = (byte) (c10 & 255);
            int i9 = this.f14233h;
            bArr[10] = (byte) ((65280 & i9) >>> 8);
            bArr[11] = (byte) (i9 & 255);
            int i10 = this.f14226a;
            if (i10 == 2) {
                bArr[4] = 2;
                bArr[5] = 32;
            } else {
                bArr[4] = 1;
                bArr[5] = 32;
            }
            for (int i11 = 0; i11 < 4; i11++) {
                try {
                    this.f14232g.write(this.f14228c, 0, 16);
                    this.f14232g.flush();
                    break;
                } catch (IOException e11) {
                    if (i11 >= 3) {
                        throw e11;
                    }
                    e11.toString();
                }
            }
            h.p(10);
            byte[] a10 = a(this.f14231f);
            if (a10 == null) {
                throw new IOException();
            }
            if (a10.length < 16) {
                throw new IOException();
            }
            if (i10 == 2) {
                e10 = h.e(f14225z, a10, 8);
                if (e10 == null) {
                    throw new IOException();
                }
            } else {
                e10 = h.e(f14224y, a10, 8);
                if (e10 == null) {
                    throw new IOException();
                }
            }
            int i12 = ((e10[14] & 255) * 256) + ((e10[13] & 255) * 65536) + ((e10[12] & 255) * 16777216) + (e10[15] & 255);
            if (i12 <= 0) {
                throw new IOException();
            }
            byte[] bArr2 = new byte[i12];
            System.arraycopy(e10, 16, bArr2, 0, i12);
            return bArr2;
        } catch (IOException e12) {
            e12.toString();
            return null;
        }
    }

    @Override // h7.InterfaceC1632b
    public final int write(byte[] bArr, int i9, int i10) {
        byte b10;
        byte[] e10;
        if (this.f14230e == null || this.f14231f == null || this.f14232g == null || this.f14233h == 0 || bArr == null) {
            return -1;
        }
        byte b11 = 1;
        if (i10 < 1) {
            return -1;
        }
        int i11 = i10 <= 4096 ? i10 : 4096;
        long currentTimeMillis = System.currentTimeMillis();
        int i12 = 0;
        while (true) {
            try {
                long currentTimeMillis2 = System.currentTimeMillis();
                if (4000 + currentTimeMillis < currentTimeMillis2) {
                    b10 = b11;
                    currentTimeMillis = currentTimeMillis2;
                } else {
                    b10 = 0;
                }
                if (b10 != 0) {
                    break;
                }
                System.arraycopy(f14219t, 0, this.f14228c, 0, 16);
                int i13 = this.f14226a;
                if (i13 == 2) {
                    byte[] bArr2 = this.f14228c;
                    bArr2[4] = 2;
                    bArr2[5] = 33;
                } else {
                    byte[] bArr3 = this.f14228c;
                    bArr3[4] = b11;
                    bArr3[5] = 33;
                }
                int c10 = c();
                byte[] bArr4 = this.f14228c;
                bArr4[8] = (byte) ((c10 & HI_BYTE_MASK) >>> 8);
                bArr4[9] = (byte) (c10 & 255);
                int i14 = this.f14233h;
                bArr4[10] = (byte) ((i14 & HI_BYTE_MASK) >>> 8);
                bArr4[11] = (byte) (i14 & 255);
                int i15 = i11 - i12;
                bArr4[12] = (byte) (((-16777216) & i15) >>> 24);
                bArr4[13] = (byte) ((16711680 & i15) >>> 16);
                bArr4[14] = (byte) ((i15 & HI_BYTE_MASK) >>> 8);
                bArr4[15] = (byte) (i15 & 255);
                System.arraycopy(bArr, i12 + i9, bArr4, 16, i15);
                for (int i16 = 0; i16 < 4; i16++) {
                    try {
                        this.f14232g.write(this.f14228c, 0, i15 + 16);
                        this.f14232g.flush();
                        break;
                    } catch (IOException e11) {
                        if (i16 >= 3) {
                            throw e11;
                        }
                    }
                }
                byte[] a10 = a(this.f14231f);
                if (a10 == null) {
                    throw new IOException();
                }
                if (i13 == 2) {
                    e10 = h.e(f14221v, a10, 8);
                    if (e10 == null) {
                        throw new IOException();
                    }
                } else {
                    e10 = h.e(f14220u, a10, 8);
                    if (e10 == null) {
                        throw new IOException();
                    }
                }
                i12 += ((e10[18] & 255) * 256) + ((e10[17] & 255) * 65536) + ((e10[16] & 255) * 16777216) + (e10[19] & 255);
                if (i12 >= i11) {
                    break;
                }
                h.p(200);
                b11 = 1;
            } catch (IOException unused) {
                return -1;
            }
        }
        return i12;
    }

    public C1631a(int i9) {
        this.f14229d = 1;
        this.f14236k = false;
        this.f14237l = 0L;
        this.f14238m = 0L;
        this.f14227b = new byte[24768];
        this.f14228c = new byte[24768];
        this.f14230e = null;
        this.f14231f = null;
        this.f14232g = null;
        this.f14233h = 0;
        this.f14234i = null;
        this.f14226a = 2;
        this.f14235j = 8612;
    }
}
