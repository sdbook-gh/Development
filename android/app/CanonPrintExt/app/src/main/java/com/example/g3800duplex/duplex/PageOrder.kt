package com.example.g3800duplex.duplex

/**
 * Page order for manual duplex (1-based page numbers).
 * Back-side order must be calibrated on a real G3800 feed path.
 */
enum class Binding {
    /** Flip on long edge (typical portrait / booklet left bind). */
    LONG_EDGE,

    /** Flip on short edge (typical calendar / top bind). */
    SHORT_EDGE,
}

object PageOrder {
    fun frontPages(pageCount: Int): List<Int> {
        require(pageCount >= 0)
        return (1..pageCount).filter { it % 2 == 1 }
    }

    /**
     * Even pages for the second pass after the user reloads paper.
     * Default: reverse even pages for long-edge (common rear-feed stack order).
     * Short-edge uses the same reverse for v1; calibrate on device if needed.
     */
    fun backPages(pageCount: Int, binding: Binding): List<Int> {
        require(pageCount >= 0)
        val evens = (1..pageCount).filter { it % 2 == 0 }
        return when (binding) {
            Binding.LONG_EDGE, Binding.SHORT_EDGE -> evens.reversed()
        }
    }
}
