package com.example.g3800duplex.print

import android.content.Context
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import com.example.g3800duplex.transport.PrintProtocol
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Generates a single-page PDF test page matching the selected paper size.
 */
class TestPageGenerator(private val context: Context) {
    fun generate(
        protocol: PrintProtocol,
        endpoint: String,
        printerName: String = "",
        paper: PrintPaperSettings = PrintPaperSettings(),
    ): File {
        val out = File(context.cacheDir, "test_page_${System.currentTimeMillis()}.pdf")
        val doc = PdfDocument()
        val w = paper.size.pdfWidthPt
        val h = paper.size.pdfHeightPt
        val pageInfo = PdfDocument.PageInfo.Builder(w, h, 1).create()
        val page = doc.startPage(pageInfo)
        val canvas = page.canvas

        val margin = (w * 0.08f).coerceAtLeast(24f).coerceAtMost(48f)
        val titlePaint = Paint().apply {
            isAntiAlias = true
            textSize = (w / 28f).coerceIn(14f, 22f)
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            color = 0xFF111111.toInt()
        }
        val bodyPaint = Paint().apply {
            isAntiAlias = true
            textSize = (w / 48f).coerceIn(9f, 12f)
            color = 0xFF222222.toInt()
        }
        val stamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())

        var y = margin + titlePaint.textSize
        canvas.drawText("G3800 Duplex — 打印测试页", margin, y, titlePaint)
        y += titlePaint.textSize + 12f
        canvas.drawText("协议: ${protocol.label}", margin, y, bodyPaint)
        y += bodyPaint.textSize + 8f
        canvas.drawText("纸张: ${paper.summary()}", margin, y, bodyPaint)
        y += bodyPaint.textSize + 8f
        canvas.drawText("打印机: ${printerName.ifBlank { "(未命名)" }}", margin, y, bodyPaint)
        y += bodyPaint.textSize + 8f
        canvas.drawText("Endpoint: ${endpoint.ifBlank { "(无)" }}", margin, y, bodyPaint)
        y += bodyPaint.textSize + 8f
        canvas.drawText("时间: $stamp", margin, y, bodyPaint)
        y += bodyPaint.textSize + 16f

        val barH = (h * 0.05f).coerceIn(24f, 40f)
        val barW = ((w - margin * 2) / 5.5f).coerceAtLeast(40f)
        val barPaint = Paint()
        val colors = intArrayOf(
            0xFFE53935.toInt(),
            0xFF1E88E5.toInt(),
            0xFF43A047.toInt(),
            0xFFFDD835.toInt(),
            0xFF000000.toInt(),
        )
        var x = margin
        for (c in colors) {
            barPaint.color = c
            canvas.drawRect(x, y, x + barW, y + barH, barPaint)
            x += barW + 8f
        }
        y += barH + 20f

        val linePaint = Paint().apply {
            color = 0xFF000000.toInt()
            strokeWidth = 2f
        }
        val lineEnd = (w - margin).coerceAtLeast(margin + 40f)
        canvas.drawLine(margin, y, lineEnd, y, linePaint)
        y += bodyPaint.textSize + 12f
        canvas.drawText(
            "若本页正常出纸，说明所选协议与纸张设置可用。",
            margin,
            y,
            bodyPaint,
        )

        doc.finishPage(page)
        FileOutputStream(out).use { doc.writeTo(it) }
        doc.close()
        return out
    }
}
