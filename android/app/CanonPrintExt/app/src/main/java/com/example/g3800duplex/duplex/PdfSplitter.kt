package com.example.g3800duplex.duplex

import android.content.Context
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import java.io.File

data class SplitResult(
    val frontPdf: File,
    val backPdf: File?,
    val frontPageNumbers: List<Int>,
    val backPageNumbers: List<Int>,
)

/**
 * Splits a document PDF into front/back temporary PDFs for two simplex jobs.
 */
class PdfSplitter(private val context: Context) {
    init {
        PDFBoxResourceLoader.init(context.applicationContext)
    }

    fun split(sourcePdf: File, binding: Binding, outDir: File = context.cacheDir): SplitResult {
        require(sourcePdf.exists()) { "PDF not found: ${sourcePdf.absolutePath}" }
        outDir.mkdirs()

        PDDocument.load(sourcePdf).use { doc ->
            val pageCount = doc.numberOfPages
            val frontIdx = PageOrder.frontPages(pageCount) // 1-based
            val backIdx = PageOrder.backPages(pageCount, binding)

            val frontFile = File(outDir, "g3800_front_${System.currentTimeMillis()}.pdf")
            writeSubset(doc, frontIdx, frontFile)

            val backFile = if (backIdx.isEmpty()) {
                null
            } else {
                File(outDir, "g3800_back_${System.currentTimeMillis()}.pdf").also {
                    writeSubset(doc, backIdx, it)
                }
            }

            return SplitResult(frontFile, backFile, frontIdx, backIdx)
        }
    }

    private fun writeSubset(source: PDDocument, oneBasedPages: List<Int>, out: File) {
        PDDocument().use { target ->
            for (pageNum in oneBasedPages) {
                val page = source.getPage(pageNum - 1)
                target.importPage(page)
            }
            target.save(out)
        }
    }
}
