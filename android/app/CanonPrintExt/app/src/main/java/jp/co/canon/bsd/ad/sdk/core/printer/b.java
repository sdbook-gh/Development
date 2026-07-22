package jp.co.canon.bsd.ad.sdk.core.printer;

import m2.AbstractC1862a;

/**
 * Slim IjPrinter stand-in for SNMP discovery callbacks.
 * Full decompiled IjPrinter pulls the entire Canon app; discovery only needs identity fields.
 */
public class b extends AbstractC1862a {
    private String productSerialnumber;
    private String deviceId;
    private int protocolSearching;
    private int protocolGettingStatus;

    public b() {
        setDeviceCategory(DEVICE_CATEGORY_IJ);
    }

    public void setProductSerialnumber(String serial) {
        this.productSerialnumber = serial;
    }

    public String getProductSerialnumber() {
        return productSerialnumber;
    }

    public void setDeviceId(String deviceId) {
        this.deviceId = deviceId;
    }

    public String getDeviceId() {
        return deviceId;
    }

    public void setProtocolSearching(int protocolSearching) {
        this.protocolSearching = protocolSearching;
    }

    public int getProtocolSearching() {
        return protocolSearching;
    }

    public void setProtocolGettingStatus(int protocolGettingStatus) {
        this.protocolGettingStatus = protocolGettingStatus;
    }

    public int getProtocolGettingStatus() {
        return protocolGettingStatus;
    }
}
