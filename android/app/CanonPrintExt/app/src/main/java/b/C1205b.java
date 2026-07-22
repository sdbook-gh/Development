package b;

import java.io.InputStream;
import java.net.URL;
import java.util.HashMap;
import java.util.Locale;
import jp.co.canon.android.cnml.common.CNMLJCmnUtil;

/* compiled from: APOHttpResponse.java */
/* renamed from: b.b, reason: case insensitive filesystem */
/* loaded from: /mnt/f/print/classes.dex */
public final class C1205b {

    /* renamed from: a, reason: collision with root package name */
    public int f9781a;

    /* renamed from: b, reason: collision with root package name */
    public HashMap<String, String> f9782b;

    /* renamed from: c, reason: collision with root package name */
    public InputStream f9783c;

    /* renamed from: d, reason: collision with root package name */
    public String f9784d;

    /* renamed from: e, reason: collision with root package name */
    public URL f9785e;

    /* renamed from: f, reason: collision with root package name */
    public String f9786f;

    /* renamed from: g, reason: collision with root package name */
    public boolean f9787g;

    public static String a(String str) {
        if (str == null || str.isEmpty()) {
            return "UTF-8";
        }
        String str2;
        if (str.contains("charset")) {
            String substring = str.substring(str.indexOf(";"), str.length());
            str2 = substring.substring(substring.indexOf(CNMLJCmnUtil.EQUAL) + 1, substring.length());
        } else {
            str2 = "UTF-8";
        }
        return str2.toUpperCase(Locale.ROOT);
    }

    public final String b(String str) {
        return this.f9782b.get(str.toUpperCase(Locale.ROOT));
    }
}
