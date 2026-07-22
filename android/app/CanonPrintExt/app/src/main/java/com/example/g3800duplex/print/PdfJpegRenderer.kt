package com.example.g3800duplex.print

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.pdf.PdfRenderer
import android.os.ParcelFileDescriptor
import java.io.File
import java.io.FileOutputStream

/**
 * Renders PDF pages to JPEG files for CLSS SendData (format JPEG/JPEGPAGE).
 */
class PdfJpegRenderer(private val context: Context) {
    fun renderAllPages(pdf: File, outDir: File = context.cacheDir, maxEdge: Int = 2480): List<File> {
        require(pdf.exists()) { "PDF missing: ${pdf.absolutePath}" }
        outDir.mkdirs()
        val results = ArrayList<File>()
        ParcelFileDescriptor.open(pdf, ParcelFileDescriptor.MODE_READ_ONLY).use { pfd ->
            PdfRenderer(pfd).use { renderer ->
                for (i in 0 until renderer.pageCount) {
                    renderer.openPage(i).use { page ->
                        val scale = maxEdge.toFloat() / maxOf(page.width, page.height).coerceAtLeast(1)
                        val w = (page.width * scale).toInt().coerceAtLeast(1)
                        val h = (page.height * scale).toInt().coerceAtLeast(1)
                        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                        val canvas = Canvas(bitmap)
                        canvas.drawColor(Color.WHITE)
                        page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_PRINT)
                        val out = File(outDir, "clss_page_${System.currentTimeMillis()}_$i.jpg")
                        FileOutputStream(out).use { fos ->
                            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, fos)
                        }
                        bitmap.recycle()
                        results.add(out)
                    }
                }
            }
        }
        return results
    }
}
