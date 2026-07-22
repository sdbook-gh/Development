package jp.co.canon.bsd.ad.sdk.core.search;

import android.os.Process;
import androidx.annotation.NonNull;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import jp.co.canon.bsd.ad.sdk.core.clss.CLSSUtility;
import jp.co.canon.bsd.ad.sdk.core.clss.CLSS_Exception;
import k7.h;
import m2.b;

/**
 * Canon SNMP printer search — JNI-backed via libsdk-core.so.
 * Method signatures must stay compatible with native RegisterNatives / FindClass.
 */
public class SnmpSearch implements b {
    private static final int IJ_DEVICE_TYPE_NUMBER = 4;
    private static final int TIMEOUT_SEARCH = 2000;

    @NonNull
    private final String mBroadcastAddress;
    private b.a mCallback;
    private boolean mCanceled;
    private int mNumSearchedPrinter = 0;
    private static final List<SnmpSearch> WORKING_INSTANCES = new ArrayList<>();
    private static final Object LOCK = new Object();

    public class SearchThread extends Thread {
        public final b.a callback;

        public SearchThread(b.a callback) {
            this.callback = callback;
        }

        @Override
        public void run() {
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND);
            SnmpSearch.this.search(this.callback);
        }
    }

    public SnmpSearch(@NonNull String broadcastAddress) {
        this.mBroadcastAddress = broadcastAddress;
    }

    private native int StartSNMPSearch(int instanceHash, String broadcastAddress, int timeoutMs);

    /** Called from native when a printer is found. Keep signature unchanged. */
    private static void setPrinter(
            int instanceHash,
            String ip,
            String mac,
            String model,
            String serial,
            String deviceId,
            int deviceType) {
        SnmpSearch owner;
        synchronized (LOCK) {
            owner = null;
            for (SnmpSearch candidate : WORKING_INSTANCES) {
                if (candidate.hashCode() == instanceHash) {
                    owner = candidate;
                    break;
                }
            }
        }
        if (owner == null) {
            return;
        }
        owner.mNumSearchedPrinter++;
        int protocol;
        try {
            protocol = CLSSUtility.getProtocol(deviceId);
        } catch (CLSS_Exception e) {
            protocol = 1;
        }
        // Official app skips protocol==1; for discovery we still surface IJ devices
        // when protocol parse fails, so the UI can list G3800 on LAN.
        if (deviceType != IJ_DEVICE_TYPE_NUMBER) {
            return;
        }
        jp.co.canon.bsd.ad.sdk.core.printer.b printer = new jp.co.canon.bsd.ad.sdk.core.printer.b();
        printer.setIpAddress(ip);
        printer.setMacAddress(mac);
        printer.setModelName(model);
        printer.setProductSerialnumber(serial);
        printer.setDeviceId(deviceId);
        printer.setNickname(printer.getModelName());
        int searching = 0;
        int gettingStatus = 0;
        if (protocol == 2) {
            searching = 1;
            gettingStatus = 2;
        } else if (protocol != 1) {
            // Unknown non-1 protocol: keep defaults (0)
        }
        printer.setProtocolSearching(searching);
        printer.setProtocolGettingStatus(gettingStatus);
        b.a callback = owner.mCallback;
        if (callback != null) {
            callback.b(printer);
        }
    }

    public boolean isWorking() {
        synchronized (LOCK) {
            return WORKING_INSTANCES.contains(this);
        }
    }

    public int search(@NonNull b.a callback) {
        this.mNumSearchedPrinter = 0;
        this.mCanceled = false;
        this.mCallback = callback;
        int expected = StartSNMPSearch(hashCode(), this.mBroadcastAddress, TIMEOUT_SEARCH);
        if (expected > 0) {
            while (!this.mCanceled && expected != this.mNumSearchedPrinter) {
                h.p(100);
            }
        }
        boolean removed;
        synchronized (LOCK) {
            removed = WORKING_INSTANCES.remove(this);
        }
        if (!removed) {
            callback.a(2);
        } else if (this.mCanceled) {
            callback.a(1);
        } else if (expected < 0) {
            callback.a(2);
        } else {
            callback.a(0);
        }
        return 0;
    }

    @Override
    public int startSearch(@NonNull b.a callback) {
        synchronized (LOCK) {
            if (!WORKING_INSTANCES.contains(this) && WORKING_INSTANCES.add(this)) {
                new SearchThread(callback).start();
                return 0;
            }
            return -1;
        }
    }

    @Override
    public int stopSearch() {
        this.mCanceled = true;
        return 0;
    }
}
