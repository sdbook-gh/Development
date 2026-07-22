package C3;

/** Minimal string builder helper used by MobileATP. */
public final class d {
    private d() {}

    public static StringBuilder c(String str) {
        return new StringBuilder(str == null ? "" : str);
    }
}
