package jp.co.canon.oip.android.opal.mobileatp.util;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Properties;
import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;

/* compiled from: FileUtil.java */
/* loaded from: /mnt/f/print/classes3.dex */
public class d {
    private d() {
    }

    public static File a(String str) {
        if (str == null) {
            throw new ATPException(1000, "app dirname is null.");
        }
        File file = new File(str);
        if (b(file) && !file.isDirectory()) {
            file.delete();
        }
        if (!b(file)) {
            file.mkdirs();
        }
        return file;
    }

    public static File b(File file, String str) {
        if (file == null) {
            throw new ATPException(1000, "getFile fileDir is null.");
        }
        if (str == null) {
            throw new ATPException(1001, "getFile filneName is null.");
        }
        return new File(c(file.getPath() + '/' + str));
    }

    public static InputStream c(File file) {
        if (file == null) {
            throw new ATPException(1003, "getFileInputStream file is null.");
        }
        try {
            return new BufferedInputStream(new FileInputStream(new File(c(file.getAbsolutePath()))));
        } catch (IOException e10) {
            throw new ATPException(1003, e10.getMessage(), e10);
        }
    }

    public static OutputStream d(File file) {
        if (file == null) {
            throw new ATPException(1004, "getFileOutputStream file is null.");
        }
        try {
            return new BufferedOutputStream(new FileOutputStream(new File(c(file.getAbsolutePath()))));
        } catch (IOException e10) {
            throw new ATPException(1004, e10.getMessage(), e10);
        }
    }

    public static boolean b(File file) {
        return file != null && file.exists();
    }

    public static Properties b(String str) {
        Properties properties = new Properties();
        if (g.a(str)) {
            return properties;
        }
        FileInputStream fileInputStream = null;
        try {
            fileInputStream = new FileInputStream(str);
            properties.load(fileInputStream);
            return properties;
        } catch (FileNotFoundException e10) {
            throw new ATPException(1003, e10.getMessage(), e10);
        } catch (IOException e11) {
            throw new ATPException(1003, e11.getMessage(), e11);
        } finally {
            if (fileInputStream != null) {
                try {
                    fileInputStream.close();
                } catch (IOException e12) {
                    jp.co.canon.oip.android.opal.mobileatp.d.b.a(e12);
                }
            }
        }
    }

    public static String c(String str) {
        if (str != null) {
            return str;
        }
        throw new ATPException(1006);
    }

    public static File a(File file, String str) {
        if (file == null) {
            throw new ATPException(1000, "createNewFile fileDir is null.");
        }
        if (str != null) {
            File file2 = new File(c(file.getPath() + '/' + str));
            if (b(file2)) {
                file2.delete();
            }
            try {
                file2.createNewFile();
                return file2;
            } catch (IOException e10) {
                throw new ATPException(1099, e10.getMessage(), e10);
            }
        }
        throw new ATPException(1001, "createNewFile filneName is null.");
    }

    public static void a(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        file.delete();
    }
}
