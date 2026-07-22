package k7;

/** Timeout helper used by BjnpSearch receive loop. */
public final class g {
    private long startMs;
    private long durationMs;

    public g(int durationMs) {
        this.startMs = System.currentTimeMillis();
        this.durationMs = durationMs;
    }

    public boolean a() {
        return this.startMs + this.durationMs < System.currentTimeMillis();
    }

    /** Returns true once when the window has expired (then resets start). */
    public boolean b() {
        long now = System.currentTimeMillis();
        if (this.startMs + this.durationMs >= now) {
            return false;
        }
        this.startMs = now;
        return true;
    }
}
