package jp.co.canon.bsd.ad.sdk.core.util.struct;

/** Minimal stub for CLSSCapabilityDeviceInfo compile. */
public class CipherData {
    private byte[] mData;
    private byte[] mParams;

    public CipherData(byte[] data, byte[] params) {
        this.mData = data;
        this.mParams = params;
    }

    public byte[] getData() {
        return this.mData;
    }

    public byte[] getParams() {
        return this.mParams;
    }

    public void setData(byte[] data) {
        this.mData = data;
    }

    public void setParams(byte[] params) {
        this.mParams = params;
    }
}
