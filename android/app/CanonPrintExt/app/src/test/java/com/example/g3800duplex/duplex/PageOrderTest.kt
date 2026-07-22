package com.example.g3800duplex.duplex

import org.junit.Assert.assertEquals
import org.junit.Test

class PageOrderTest {
    @Test
    fun frontPages_oddAndEvenCounts() {
        assertEquals(listOf(1, 3, 5), PageOrder.frontPages(5))
        assertEquals(listOf(1, 3, 5), PageOrder.frontPages(6))
    }

    @Test
    fun backPages_longEdgeReversedEvens() {
        assertEquals(listOf(6, 4, 2), PageOrder.backPages(6, Binding.LONG_EDGE))
        assertEquals(listOf(4, 2), PageOrder.backPages(5, Binding.LONG_EDGE))
        assertEquals(emptyList<Int>(), PageOrder.backPages(1, Binding.LONG_EDGE))
    }
}
