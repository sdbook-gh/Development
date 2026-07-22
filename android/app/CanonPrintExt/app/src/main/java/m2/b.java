package m2;

/**
 * Minimal stub of Canon SearchPrinter interface (compiled from SearchPrinter.java).
 */
public interface b {

    interface a {
        /** Search finished: 0=ok, 1=canceled, 2=error */
        void a(int resultCode);

        /** Printer found */
        void b(AbstractC1862a printer);
    }

    int startSearch(a callback);

    int stopSearch();
}
