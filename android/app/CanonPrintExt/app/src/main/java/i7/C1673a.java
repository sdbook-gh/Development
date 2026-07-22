package i7;

import android.os.Process;
import android.util.Log;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.util.Arrays;
import jp.co.canon.bsd.ad.sdk.core.printer.b;
import jp.co.canon.bsd.ad.sdk.core.printer.j;
import k7.g;
import k7.h;
import m2.AbstractC1862a;

/**
 * BJNP UDP discovery on port 8611 (Canon BjnpSearch / i7.C1673a, cleaned).
 */
public final class C1673a implements m2.b {
    private static final String TAG = "G3800Bjnp";
    private static final byte[] DISCOVER_REQ = {
            66, 74, 78, 80, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    private static final byte[] DISCOVER_RESP = {66, 74, 78, 80, -127, 1, 0, 0};
    private static final int RECEIVE_WINDOW_MS = 3000;

    private byte[] recvBuf = new byte[4096];
    private volatile boolean canceled;
    private boolean working;
    private final String broadcastAddress;

    public C1673a(String broadcastAddress) {
        this.broadcastAddress = broadcastAddress;
    }

    private final class SearchThread extends Thread {
        private final m2.b.a callback;

        SearchThread(m2.b.a callback) {
            this.callback = callback;
        }

        @Override
        public void run() {
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND);
            DatagramSocket socket = null;
            int finishCode = 0;
            try {
                Log.i(TAG, "[discover] broadcast → " + C1673a.this.broadcastAddress + ":8611");
                socket = new DatagramSocket();
                socket.setSoTimeout(200);
                socket.setBroadcast(true);
                socket.send(new DatagramPacket(
                        DISCOVER_REQ,
                        DISCOVER_REQ.length,
                        InetAddress.getByName(C1673a.this.broadcastAddress),
                        8611));

                g window = new g(RECEIVE_WINDOW_MS);
                int hits = 0;
                while (!window.b()) {
                    if (C1673a.this.canceled) {
                        finishCode = 1;
                        break;
                    }
                    h.p(100);
                    int got = receiveDiscover(socket);
                    if (got <= 0) {
                        continue;
                    }
                    byte[] ipBytes = Arrays.copyOfRange(C1673a.this.recvBuf, 28, 32);
                    byte[] macBytes = Arrays.copyOfRange(C1673a.this.recvBuf, 22, 28);
                    String ip = h.g(ipBytes);
                    String mac = h.h(macBytes);
                    if (ip == null || mac == null) {
                        continue;
                    }
                    AbstractC1862a printer = buildPrinter(ip, mac);
                    if (printer != null) {
                        hits++;
                        Log.i(TAG, "[discover] hit#" + hits + " ip=" + ip
                                + " mac=" + mac
                                + " model=" + printer.getModelName());
                        this.callback.b(printer);
                    } else {
                        Log.d(TAG, "[discover] skip non-IJ/Canon ip=" + ip + " mac=" + mac);
                    }
                }
                Log.i(TAG, "[discover] done hits=" + hits + " finishCode=" + finishCode);
            } catch (IOException e) {
                finishCode = C1673a.this.canceled ? 1 : 2;
                Log.e(TAG, "[discover] IOException finishCode=" + finishCode
                        + " " + e.getClass().getSimpleName() + ": " + e.getMessage());
            } finally {
                if (socket != null) {
                    try {
                        socket.close();
                    } catch (Exception ignored) {
                    }
                }
                synchronized (C1673a.this) {
                    C1673a.this.working = false;
                }
                if (C1673a.this.canceled && finishCode == 0) {
                    finishCode = 1;
                }
                this.callback.a(finishCode);
            }
        }
    }

    private AbstractC1862a buildPrinter(String ip, String mac) {
        String deviceId = C1674b.a(ip);
        if (deviceId == null || !h.n(deviceId, "MFG", "Canon")) {
            return null;
        }
        // Inkjet G-series: prefer CMD containing IVEC; still accept Canon with MDL if CMD absent.
        boolean hasIvec = h.n(deviceId, "CMD", "IVEC");
        String model = h.m(deviceId, "MDL");
        if (model == null || model.isEmpty()) {
            return null;
        }
        if (!hasIvec && h.m(deviceId, "CMD") != null && !h.n(deviceId, "CMD", "IVEC")) {
            // Non-IVEC CMD present (e.g. other Canon families) — skip for IJ search.
            return null;
        }

        b printer = new b();
        printer.setIpAddress(ip);
        printer.setMacAddress(mac);
        printer.setDeviceId(deviceId);
        printer.setModelName(model);
        printer.setNickname(model);
        int protocol = j.c(deviceId);
        if (protocol == 2) {
            printer.setProtocolSearching(1);
            printer.setProtocolGettingStatus(2);
        } else {
            // Soft fallback (same as SnmpSearch path) — still list the printer.
            printer.setProtocolSearching(0);
            printer.setProtocolGettingStatus(0);
        }
        return printer;
    }

    /**
     * Receive one BJNP discover response into recvBuf. Returns payload size or -1.
     */
    private int receiveDiscover(DatagramSocket socket) {
        if (this.canceled) {
            return -1;
        }
        try {
            DatagramPacket packet = new DatagramPacket(this.recvBuf, this.recvBuf.length);
            socket.receive(packet);
            int len = packet.getLength();
            if (len < 32) {
                return -1;
            }
            byte[] matched = h.e(DISCOVER_RESP, this.recvBuf, 8);
            if (matched == null || matched.length < 32) {
                return -1;
            }
            // Keep response aligned at buffer start for IP/MAC offsets.
            System.arraycopy(matched, 0, this.recvBuf, 0, Math.min(matched.length, this.recvBuf.length));
            int payloadLen =
                    ((matched[14] & 255) * 256)
                            + ((matched[13] & 255) * 65536)
                            + ((matched[12] & 255) * 16777216)
                            + (matched[15] & 255);
            return payloadLen;
        } catch (IOException unused) {
            return -1;
        }
    }

    @Override
    public int startSearch(m2.b.a callback) {
        synchronized (this) {
            if (this.working) {
                return -1;
            }
            this.working = true;
            this.canceled = false;
            new SearchThread(callback).start();
            return 0;
        }
    }

    @Override
    public int stopSearch() {
        this.canceled = true;
        return 0;
    }
}
