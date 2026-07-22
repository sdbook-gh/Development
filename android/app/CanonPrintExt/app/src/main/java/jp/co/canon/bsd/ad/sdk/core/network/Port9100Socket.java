package jp.co.canon.bsd.ad.sdk.core.network;

import h7.InterfaceC1632b;

/* loaded from: /mnt/f/print/classes3.dex */
public class Port9100Socket implements InterfaceC1632b {
    private long mWorkAddress;

    public native boolean ClosePort9100();

    public native boolean OpenPort9100(String str);

    public native int WritePort9100(byte[] bArr, int i9);

    @Override // h7.InterfaceC1632b
    public void close() {
        ClosePort9100();
    }

    @Override // h7.InterfaceC1632b
    public int open(String str) {
        return OpenPort9100(str) ? 0 : -4;
    }

    @Override // h7.InterfaceC1632b
    public byte[] read() {
        throw new UnsupportedOperationException();
    }

    @Override // h7.InterfaceC1632b
    public int write(byte[] bArr, int i9, int i10) {
        return WritePort9100(bArr, i10);
    }
}
