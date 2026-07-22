package b;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.Proxy;
import java.net.URL;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Minimal APOHttpClient stand-in for MobileATP (HttpURLConnection).
 * Implements Digests challenge-response like the official client (no Basic-first).
 */
public final class C1204a {

    public int f9762a; // connect timeout
    public int f9763b; // so timeout
    public ByteArrayOutputStream f9764c;
    public InputStream f9765d;
    public String f9766e;
    public String f9767f; // digest username
    public String f9768g; // digest password
    public String f9769h;
    public int f9770i;
    public Object f9771j;
    public int f9772k;
    public String f9773l;
    public String f9774m;
    public HashMap<String, String> f9775n;
    public Object f9776o;
    public String f9777p;
    public int f9778q;
    public Object f9779r;
    public Object f9780s;

    private HttpURLConnection connection;

    public void d(String name, String value) {
        if (f9775n == null) {
            f9775n = new HashMap<>();
        }
        f9775n.put(name, value);
    }

    public C1205b a(URL url) throws IOException {
        C1205b first = execute(url, null);
        if (first.f9781a == 401
                && f9767f != null
                && !f9767f.isEmpty()
                && (f9764c != null || f9765d != null
                || "GET".equalsIgnoreCase(f9766e)
                || "DELETE".equalsIgnoreCase(f9766e))) {
            String auth = new DigestAuth(f9767f, f9768g).authorizationHeader(first);
            if (auth != null && !auth.isEmpty()) {
                if (f9764c != null) {
                    f9765d = new ByteArrayInputStream(f9764c.toByteArray());
                }
                e();
                return execute(url, auth);
            }
        }
        return first;
    }

    private C1205b execute(URL url, String authorization) throws IOException {
        Proxy proxy = Proxy.NO_PROXY;
        if (f9769h != null && !f9769h.isEmpty() && f9770i > 0) {
            proxy = new Proxy(Proxy.Type.HTTP, new InetSocketAddress(f9769h, f9770i));
        }
        connection = (HttpURLConnection) url.openConnection(proxy);
        connection.setConnectTimeout(Math.max(f9762a, 1));
        connection.setReadTimeout(Math.max(f9763b, 1));
        connection.setInstanceFollowRedirects(false);
        String method = f9766e == null || f9766e.isEmpty() ? "GET" : f9766e;
        connection.setRequestMethod(method);
        connection.setDoInput(true);

        if (f9777p != null && !f9777p.isEmpty()) {
            connection.setRequestProperty("Proxy-Authorization", f9777p);
        }
        if (authorization != null && !authorization.isEmpty()) {
            connection.setRequestProperty("Authorization", authorization);
        }
        if (f9775n != null) {
            for (Map.Entry<String, String> e : f9775n.entrySet()) {
                if (e.getKey() != null && e.getValue() != null) {
                    connection.setRequestProperty(e.getKey(), e.getValue());
                }
            }
        }

        byte[] body = null;
        if (f9765d != null) {
            body = readAll(f9765d);
            // Allow a second read after 401 → Digest retry.
            if (f9764c != null) {
                f9765d = new ByteArrayInputStream(f9764c.toByteArray());
            }
        } else if (f9764c != null) {
            body = f9764c.toByteArray();
        }
        if (body != null) {
            connection.setDoOutput(true);
            if (f9775n == null || !containsIgnoreCase(f9775n, "Content-Length")) {
                connection.setFixedLengthStreamingMode(body.length);
            }
            try (OutputStream os = connection.getOutputStream()) {
                os.write(body);
                os.flush();
            }
        }

        int code = connection.getResponseCode();
        String contentType = connection.getContentType();
        InputStream stream;
        try {
            stream = code >= 400 ? connection.getErrorStream() : connection.getInputStream();
        } catch (IOException ignored) {
            stream = connection.getErrorStream();
        }
        if (stream == null) {
            stream = new ByteArrayInputStream(new byte[0]);
        }
        byte[] responseBytes = readAll(stream);
        C1205b response = new C1205b();
        response.f9781a = code;
        response.f9783c = new ByteArrayInputStream(responseBytes);
        response.f9784d = contentType == null ? "application/json;charset=UTF-8" : contentType;
        response.f9785e = url;
        response.f9786f = method;
        response.f9787g = "https".equalsIgnoreCase(url.getProtocol());
        response.f9782b = new HashMap<>();
        Map<String, java.util.List<String>> headers = connection.getHeaderFields();
        if (headers != null) {
            for (Map.Entry<String, java.util.List<String>> e : headers.entrySet()) {
                if (e.getKey() == null || e.getValue() == null || e.getValue().isEmpty()) {
                    continue;
                }
                response.f9782b.put(e.getKey().toUpperCase(Locale.ROOT), e.getValue().get(0));
            }
        }
        return response;
    }

    public void e() {
        if (connection != null) {
            connection.disconnect();
            connection = null;
        }
    }

    private static boolean containsIgnoreCase(HashMap<String, String> map, String key) {
        for (String k : map.keySet()) {
            if (k != null && k.equalsIgnoreCase(key)) {
                return true;
            }
        }
        return false;
    }

    private static byte[] readAll(InputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) != -1) {
            out.write(buf, 0, n);
        }
        return out.toByteArray();
    }
}
