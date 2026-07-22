package b;

import android.util.Base64;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Locale;

/**
 * HTTP Digest/Basic challenge response for Canon CAMS (ported from official APO auth).
 * Server auth (401) requires algorithm=SHA-512 and qop=auth.
 */
final class DigestAuth {

    private final String username;
    private final String password;
    private String algorithm;

    DigestAuth(String username, String password) {
        this.username = username == null ? "" : username;
        this.password = password == null ? "" : password;
    }

    String authorizationHeader(C1205b challengeResponse) throws IOException {
        if (challengeResponse == null || challengeResponse.f9781a != 401) {
            return "";
        }
        String header = challengeResponse.b("WWW-Authenticate");
        if (header == null || header.isEmpty()) {
            return "";
        }
        if (header.startsWith("Basic ")) {
            String token = Base64.encodeToString(
                    (username + ":" + password).getBytes(StandardCharsets.UTF_8),
                    Base64.NO_WRAP);
            return "Basic " + token;
        }
        if (!header.startsWith("Digest ")) {
            throw new IOException("Unrecognized authentication challenge");
        }
        try {
            HashMap<String, String> params = parseChallenge(header.substring("Digest ".length()));
            params.put("username", username);
            params.put("password", password);
            params.put("method", challengeResponse.f9786f);
            String path = challengeResponse.f9785e != null ? challengeResponse.f9785e.getPath() : "";
            if (path == null || path.isEmpty()) {
                path = "/";
            }
            validateDigestParams(params);
            String realm = params.get("realm");
            params.put("uri", path + "?realm=" + URLEncoder.encode(realm, "UTF-8"));
            params.put("nc", "00000001");
            params.put("cnonce", cnonce());
            return buildDigestHeader(params);
        } catch (IllegalArgumentException | IllegalStateException | NoSuchAlgorithmException e) {
            throw new IOException(e.getMessage(), e);
        }
    }

    private static HashMap<String, String> parseChallenge(String challenge) {
        String[] parts = challenge.split(",");
        HashMap<String, String> map = new HashMap<>((int) ((parts.length / 0.75d) + 1.0d));
        Locale locale = Locale.ENGLISH;
        for (String part : parts) {
            String trim = part.replace("\"", "").trim();
            int eq = trim.indexOf('=');
            if (eq <= 0) {
                continue;
            }
            String key = trim.substring(0, eq).toLowerCase(locale);
            String value = trim.substring(eq + 1);
            map.put(key, value);
        }
        return map;
    }

    private void validateDigestParams(HashMap<String, String> params) {
        if (params.get("realm") == null) {
            throw new IllegalStateException("Realm should not be null");
        }
        if (params.get("nonce") == null) {
            throw new IllegalStateException("Nonce should not be null");
        }
        String algo = params.get("algorithm");
        if (algo == null) {
            throw new IllegalArgumentException("No Hash Algorithm Specified");
        }
        if (!algo.equalsIgnoreCase("SHA-512")) {
            throw new IllegalArgumentException("Unsupported Hash Algorithm Specified");
        }
        this.algorithm = algo;
        String qop = params.get("qop");
        if (qop == null) {
            throw new IllegalArgumentException("No QOP directive specified in authentication challenge");
        }
        String selected = null;
        Iterator<String> it = Arrays.asList(qop.trim().split("\\s*,\\s*")).iterator();
        while (it.hasNext()) {
            String item = it.next();
            if (item.equalsIgnoreCase("auth")) {
                selected = item;
                break;
            }
        }
        if (selected == null) {
            throw new IllegalArgumentException("Unsupported QOP directive specified in authentication challenge");
        }
        params.put("qop", selected);
    }

    private String cnonce() throws NoSuchAlgorithmException {
        String stamp = new SimpleDateFormat("EEE yyyy MM dd HH mm ss z", Locale.ENGLISH).format(new Date());
        return hexDigest(stamp);
    }

    private String buildDigestHeader(HashMap<String, String> params) throws NoSuchAlgorithmException {
        String ha1 = hexDigest(params.get("username") + ":" + params.get("realm") + ":" + params.get("password"));
        String ha2 = hexDigest(params.get("method") + ":" + params.get("uri"));
        String response = hexDigest(
                ha1 + ":"
                        + params.get("nonce") + ":"
                        + params.get("nc") + ":"
                        + params.get("cnonce") + ":"
                        + params.get("qop") + ":"
                        + ha2);
        StringBuilder sb = new StringBuilder("Digest username=\"");
        sb.append(params.get("username"));
        sb.append("\", realm=\"");
        sb.append(params.get("realm"));
        sb.append("\", nonce=\"");
        sb.append(params.get("nonce"));
        sb.append("\", uri=\"");
        sb.append(params.get("uri"));
        sb.append("\", response=\"");
        sb.append(response);
        sb.append("\", qop=");
        sb.append(params.get("qop"));
        sb.append(", nc=");
        sb.append(params.get("nc"));
        sb.append(", cnonce=\"");
        sb.append(params.get("cnonce"));
        sb.append("\", algorithm=");
        sb.append(algorithm);
        if (params.get("opaque") != null) {
            sb.append(", opaque=\"");
            sb.append(params.get("opaque"));
            sb.append('"');
        }
        return sb.toString();
    }

    private String hexDigest(String value) throws NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance(algorithm);
        byte[] dig = md.digest(value.getBytes(StandardCharsets.UTF_8));
        StringBuilder sb = new StringBuilder(dig.length * 2);
        for (byte b : dig) {
            sb.append(Integer.toString((b & 0xff) + 0x100, 16).substring(1));
        }
        return sb.toString();
    }
}
