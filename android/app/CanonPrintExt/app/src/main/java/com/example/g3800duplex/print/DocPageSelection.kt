package com.example.g3800duplex.print

import android.content.Context
import android.graphics.pdf.PdfRenderer
import android.os.ParcelFileDescriptor
import com.example.g3800duplex.canon.NormalizedDoc
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import java.io.File

/**
 * Page count / subset helpers for print page selection (1-based page numbers).
 */
object DocPageSelection {
    fun pageCount(context: Context, doc: NormalizedDoc): Int = when (doc) {
        is NormalizedDoc.JpegPages -> doc.pages.size
        is NormalizedDoc.LocalPdf -> pdfPageCount(doc.pdf)
    }

    fun subset(
        context: Context,
        doc: NormalizedDoc,
        oneBasedPages: Collection<Int>,
    ): NormalizedDoc {
        val ordered = oneBasedPages.filter { it >= 1 }.distinct().sorted()
        require(ordered.isNotEmpty()) { "请至少选择一页" }
        val total = pageCount(context, doc)
        require(ordered.all { it <= total }) { "页码超出范围 1–$total: $ordered" }

        return when (doc) {
            is NormalizedDoc.JpegPages -> {
                val pages = ordered.map { doc.pages[it - 1] }
                NormalizedDoc.JpegPages(pages, "${doc.sourceName}（选 ${ordered.size} 页）")
            }
            is NormalizedDoc.LocalPdf -> {
                PDFBoxResourceLoader.init(context.applicationContext)
                val out = File(
                    context.cacheDir,
                    "selected_pages_${System.currentTimeMillis()}.pdf",
                )
                PDDocument.load(doc.pdf).use { source ->
                    PDDocument().use { target ->
                        for (pageNum in ordered) {
                            target.importPage(source.getPage(pageNum - 1))
                        }
                        target.save(out)
                    }
                }
                NormalizedDoc.LocalPdf(out)
            }
        }
    }

    /** Parse "1-3,5,8-9" → sorted unique 1-based pages within [1, total]. */
    fun parseRange(input: String, total: Int): Set<Int> {
        if (total < 1) return emptySet()
        val result = linkedSetOf<Int>()
        val parts = input.split(',', '，', ';', '；', ' ')
            .map { it.trim() }
            .filter { it.isNotEmpty() }
        for (part in parts) {
            val range = part.split('-', '–', '—')
            when (range.size) {
                1 -> {
                    val n = range[0].toIntOrNull() ?: continue
                    if (n in 1..total) result += n
                }
                2 -> {
                    val a = range[0].toIntOrNull() ?: continue
                    val b = range[1].toIntOrNull() ?: continue
                    val lo = minOf(a, b).coerceAtLeast(1)
                    val hi = maxOf(a, b).coerceAtMost(total)
                    for (n in lo..hi) result += n
                }
            }
        }
        return result
    }

    fun summary(selected: Set<Int>, total: Int): String {
        if (total <= 0) return "无页"
        if (selected.isEmpty()) return "未选页（共 $total 页）"
        if (selected.size == total) return "全部 $total 页"
        val ordered = selected.sorted()
        return "已选 ${ordered.size}/$total 页：${formatRange(selected)}"
    }

    /** Compact selected pages for the range text field, e.g. `1-3,5`. */
    fun formatRange(selected: Collection<Int>): String =
        compactRanges(selected.filter { it >= 1 }.distinct().sorted())

    private fun compactRanges(ordered: List<Int>): String {
        if (ordered.isEmpty()) return ""
        val parts = mutableListOf<String>()
        var start = ordered[0]
        var prev = ordered[0]
        for (i in 1 until ordered.size) {
            val n = ordered[i]
            if (n == prev + 1) {
                prev = n
                continue
            }
            parts += if (start == prev) "$start" else "$start-$prev"
            start = n
            prev = n
        }
        parts += if (start == prev) "$start" else "$start-$prev"
        return parts.joinToString(",")
    }

    private fun pdfPageCount(pdf: File): Int {
        ParcelFileDescriptor.open(pdf, ParcelFileDescriptor.MODE_READ_ONLY).use { pfd ->
            PdfRenderer(pfd).use { return it.pageCount }
        }
    }
}
