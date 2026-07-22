package com.example.g3800duplex.print

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream

/**
 * Decode gallery images (with EXIF orientation), fit onto selected paper canvas, write JPEGs for CLSS.
 */
class ImageJpegPrep(private val context: Context) {
    fun urisToJpegPages(
        uris: List<Uri>,
        paper: PrintPaperSettings,
        outDir: File = File(context.cacheDir, "album_jpeg_${System.currentTimeMillis()}"),
        maxEdge: Int = 2480,
        jpegQuality: Int = 90,
    ): List<File> {
        require(uris.isNotEmpty()) { "未选择图片" }
        outDir.mkdirs()
        val pageW = paper.size.pdfWidthPt
        val pageH = paper.size.pdfHeightPt
        val scale = maxEdge.toFloat() / maxOf(pageW, pageH).coerceAtLeast(1)
        val canvasW = (pageW * scale).toInt().coerceAtLeast(1)
        val canvasH = (pageH * scale).toInt().coerceAtLeast(1)

        val results = ArrayList<File>(uris.size)
        uris.forEachIndexed { index, uri ->
            val src = decodeOrientedBitmap(uri, maxEdge)
                ?: throw IllegalArgumentException("无法解码图片: $uri")
            try {
                val page = Bitmap.createBitmap(canvasW, canvasH, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(page)
                canvas.drawColor(Color.WHITE)
                val dst = fitContain(src.width, src.height, canvasW, canvasH)
                val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
                canvas.drawBitmap(src, null, dst, paint)
                val out = File(outDir, "album_page_${System.currentTimeMillis()}_$index.jpg")
                FileOutputStream(out).use { fos ->
                    if (!page.compress(Bitmap.CompressFormat.JPEG, jpegQuality, fos)) {
                        throw IllegalStateException("JPEG 压缩失败 index=$index")
                    }
                }
                page.recycle()
                results.add(out)
                Log.i(TAG, "page#$index ${src.width}x${src.height} → ${canvasW}x${canvasH} ${out.name}")
            } finally {
                src.recycle()
            }
        }
        return results
    }

    private fun decodeOrientedBitmap(uri: Uri, maxEdge: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        val firstIn = context.contentResolver.openInputStream(uri)
        if (firstIn == null) {
            Log.w(TAG, "decodeOrientedBitmap: return null (openInputStream bounds) uri=$uri")
            return null
        }
        firstIn.use {
            BitmapFactory.decodeStream(it, null, bounds)
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            Log.w(TAG, "decodeOrientedBitmap: return null (invalid bounds w=${bounds.outWidth} h=${bounds.outHeight}) uri=$uri")
            return null
        }

        var sample = 1
        var w = bounds.outWidth
        var h = bounds.outHeight
        while (maxOf(w, h) / sample > maxEdge * 2) {
            sample *= 2
        }
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val secondIn = context.contentResolver.openInputStream(uri)
        if (secondIn == null) {
            Log.w(TAG, "decodeOrientedBitmap: return null (openInputStream decode) uri=$uri")
            return null
        }
        val decoded = secondIn.use {
            BitmapFactory.decodeStream(it, null, opts)
        }
        if (decoded == null) {
            Log.w(TAG, "decodeOrientedBitmap: return null (BitmapFactory.decodeStream) uri=$uri sample=$sample")
            return null
        }

        val orientation = try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                ExifInterface(input).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL,
                )
            } ?: ExifInterface.ORIENTATION_NORMAL
        } catch (_: Throwable) {
            ExifInterface.ORIENTATION_NORMAL
        }

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.preScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f)
                matrix.preScale(-1f, 1f)
            }
            else -> Unit
        }
        if (matrix.isIdentity) return decoded
        return try {
            val rotated = Bitmap.createBitmap(
                decoded, 0, 0, decoded.width, decoded.height, matrix, true,
            )
            if (rotated != decoded) decoded.recycle()
            rotated
        } catch (_: Throwable) {
            decoded
        }
    }

    /** Center image in page with letterbox (contain). */
    private fun fitContain(srcW: Int, srcH: Int, dstW: Int, dstH: Int): RectF {
        val scale = minOf(dstW.toFloat() / srcW, dstH.toFloat() / srcH)
        val w = srcW * scale
        val h = srcH * scale
        val left = (dstW - w) / 2f
        val top = (dstH - h) / 2f
        return RectF(left, top, left + w, top + h)
    }

    companion object {
        private const val TAG = "G3800Album"
    }
}
