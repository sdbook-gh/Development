package m2;

import android.content.Context;
import android.content.SharedPreferences;
import java.util.List;
import java.util.Locale;

/** Minimal Printer base (compiled from Printer.java / m2.a). */
public abstract class AbstractC1862a {
    public static final int DEVICE_CATEGORY_IJ = 1;
    public static final int DEVICE_CATEGORY_UNKNOWN = 0;

    private int mDeviceCategory;
    private String mIpAddress;
    private String mMacAddress;
    private String mModelName;
    private String mNickname;
    private String mNsdMdl;
    private String mNsdUuid;

    public int getDeviceCategory() {
        return mDeviceCategory;
    }

    public String getIpAddress() {
        return mIpAddress;
    }

    public String getMacAddress() {
        return mMacAddress;
    }

    public String getModelName() {
        return mModelName;
    }

    public String getNickname() {
        return mNickname;
    }

    public String getNsdMdl() {
        return mNsdMdl;
    }

    public String getNsdUuid() {
        return mNsdUuid;
    }

    public void setDeviceCategory(int category) {
        mDeviceCategory = category;
    }

    public void setIpAddress(String ipAddress) {
        mIpAddress = ipAddress;
    }

    public void setMacAddress(String macAddress) {
        if (macAddress != null) {
            macAddress = macAddress.toUpperCase(Locale.getDefault());
        }
        mMacAddress = macAddress;
    }

    public void setModelName(String modelName) {
        mModelName = modelName;
    }

    public void setNickname(String nickname) {
        mNickname = nickname;
    }

    public void setNsdMdl(String nsdMdl) {
        mNsdMdl = nsdMdl;
    }

    public void setNsdUuid(String nsdUuid) {
        mNsdUuid = nsdUuid;
    }

    public void load(SharedPreferences prefs) {
        // no-op for discovery
    }

    public void save(SharedPreferences prefs) {
        // no-op for discovery
    }

    public List<c> getSettings(Context context, int functionId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !getClass().equals(obj.getClass())) {
            return false;
        }
        String otherMac = ((AbstractC1862a) obj).getMacAddress();
        String mac = mMacAddress;
        if (mac == null || otherMac == null) {
            return false;
        }
        return mac.equalsIgnoreCase(otherMac);
    }

    @Override
    public int hashCode() {
        String mac = mMacAddress;
        return mac != null ? mac.toLowerCase(Locale.ENGLISH).hashCode() : 0;
    }
}
