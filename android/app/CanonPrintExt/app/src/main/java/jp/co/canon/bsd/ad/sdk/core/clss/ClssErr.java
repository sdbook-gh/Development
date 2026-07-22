package jp.co.canon.bsd.ad.sdk.core.clss;

final class ClssErr {
    private ClssErr() {
    }

    static String c(int code, String prefix, StringBuilder sb) {
        sb.append(prefix);
        sb.append(code);
        return sb.toString();
    }
}
