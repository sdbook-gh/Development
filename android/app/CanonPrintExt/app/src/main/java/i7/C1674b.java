package i7;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import k7.h;

/**
 * BJNP UDP probe for IEEE1284 device ID (port 8611).
 * Cleaned from Canon BjnpUdp / i7.C1674b.
 */
public final class C1674b {
    private static final byte[] REQ = {
            66, 74, 78, 80, 1, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0
    };
    private static final byte[] RESP_PREFIX = {66, 74, 78, 80, -127, 48, 0, 0};

    private C1674b() {
    }

    public static String a(String ip) {
        if (ip == null) {
            return null;
        }
        DatagramSocket socket = null;
        String deviceId = null;
        try {
            socket = new DatagramSocket();
            socket.setSoTimeout(1000);
            byte[] buf = new byte[4096];
            socket.send(new DatagramPacket(REQ, REQ.length, InetAddress.getByName(ip), 8611));
            h.p(100);
            for (int attempt = 0; attempt < 5; attempt++) {
                try {
                    DatagramPacket recv = new DatagramPacket(buf, buf.length);
                    socket.receive(recv);
                    byte[] matched = h.e(RESP_PREFIX, recv.getData(), 8);
                    if (matched == null) {
                        break;
                    }
                    int payloadLen =
                            ((matched[14] & 255) * 256)
                                    + ((matched[13] & 255) * 65536)
                                    + ((matched[12] & 255) * 16777216)
                                    + (matched[15] & 255);
                    if (payloadLen <= 2 || payloadLen > matched.length - 1 || payloadLen > 1024) {
                        break;
                    }
                    int idLen = payloadLen - 2;
                    deviceId = new String(matched, 18, idLen, "US-ASCII");
                    break;
                } catch (IOException unused) {
                    h.o();
                }
            }
        } catch (IOException unused) {
            // ignore
        } finally {
            if (socket != null) {
                try {
                    socket.close();
                } catch (Exception ignored) {
                }
            }
        }
        return deviceId;
    }
}
