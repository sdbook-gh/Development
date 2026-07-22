package jp.co.canon.oip.android.opal.mobileatp.c;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Properties;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;

/* compiled from: ATPPropertyFile.java */
public class d extends a {
    public d(String str, String str2) {
        super(str, str2);
    }

    public void a(Properties properties) {
        a(c(), properties);
    }

    public Properties d() {
        try {
            return (Properties) b(c());
        } catch (ClassCastException e10) {
            throw new ATPException(102, e10.getMessage(), e10);
        }
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.c.a
    public Object a(File file) {
        Properties properties = new Properties();
        InputStream inputStream = null;
        try {
            if (!jp.co.canon.oip.android.opal.mobileatp.util.d.b(file)) {
                return new Properties();
            }
            inputStream = jp.co.canon.oip.android.opal.mobileatp.util.d.c(file);
            properties.load(inputStream);
            return properties;
        } catch (IOException e10) {
            throw new ATPException(1003, e10.getMessage(), e10);
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException unused) {
                }
            }
        }
    }

    @Override // jp.co.canon.oip.android.opal.mobileatp.c.a
    public void a(File file, Object obj) {
        OutputStream outputStream = null;
        try {
            Properties properties = (Properties) obj;
            outputStream = jp.co.canon.oip.android.opal.mobileatp.util.d.d(file);
            properties.store(outputStream, (String) null);
        } catch (IOException e10) {
            throw new ATPException(1004, e10.getMessage(), e10);
        } finally {
            if (outputStream != null) {
                try {
                    outputStream.close();
                } catch (IOException unused) {
                }
            }
        }
    }
}
