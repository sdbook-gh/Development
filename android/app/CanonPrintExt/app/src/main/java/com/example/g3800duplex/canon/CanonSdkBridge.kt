package com.example.g3800duplex.canon

import android.content.Context
import android.net.Uri
import com.example.g3800duplex.cloud.CanonCloudDocConverter
import com.example.g3800duplex.cloud.CloudConvertException
import com.example.g3800duplex.cloud.CloudLog
import com.example.g3800duplex.cloud.DocConvertAcceptance
import com.example.g3800duplex.cloud.PrintTicketFactory
import com.example.g3800duplex.print.ClssBjnpJpegSession
import com.example.g3800duplex.print.ImageJpegPrep
import com.example.g3800duplex.print.PdfJpegRenderer
import com.example.g3800duplex.print.PrintPaperSettings
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class DiscoveredPrinter(
    val name: String,
    val model: String,
    val ipAddress: String,
    val macAddress: String = "",
    val deviceId: String = "",
    val serial: String = "",
    val source: DiscoverySource = DiscoverySource.Snmp,
    /** Opaque handle for future print jobs (IjPrinter slim). */
    val raw: Any? = null,
    /** Which transport discovered / will use this printer. */
    val protocolLabel: String = "",
    /** e.g. ipp://192.168.1.10:631/ipp/print */
    val ippUri: String = "",
    /** Android PrintService display name */
    val printServiceName: String = "",
    /** Android PrintService component "pkg/class" */
    val printServiceComponent: String = "",
)

sealed class PrintJobResult {
    data object Success : PrintJobResult()
    data class Failed(val message: String, val cause: Throwable? = null) : PrintJobResult()
}

data class InitNativeResult(
    val ok: Boolean,
    val message: String,
)

sealed class NormalizedDoc {
    data class LocalPdf(val pdf: File) : NormalizedDoc()
    /** JPEG pages from cloud Word convert or album photos. */
    data class JpegPages(val pages: List<File>, val sourceName: String) : NormalizedDoc()
}

interface CanonSdkBridge {
    fun initNative(): InitNativeResult
    suspend fun discoverPrinters(timeoutMs: Long = 10_000): List<DiscoveredPrinter>
    suspend fun printSimplexPdf(
        printer: DiscoveredPrinter,
        pdf: File,
        jobName: String,
        paper: PrintPaperSettings = PrintPaperSettings(),
    ): PrintJobResult

    suspend fun printJpegs(
        printer: DiscoveredPrinter,
        jpegs: List<File>,
        jobName: String,
        paper: PrintPaperSettings = PrintPaperSettings(),
    ): PrintJobResult

    /**
     * PDF → local render; .doc/.docx → Canon cloud JPEG pages.
     * @throws CloudConvertException / IllegalStateException when Word ToS not accepted.
     */
    suspend fun normalizeToPrintable(
        uri: Uri,
        displayName: String?,
        paper: PrintPaperSettings = PrintPaperSettings(),
        onCloudProgress: (CanonCloudDocConverter.Progress) -> Unit = {},
    ): NormalizedDoc

    /** Album / gallery images → JPEG pages sized for [paper]. */
    suspend fun normalizeImages(
        uris: List<Uri>,
        paper: PrintPaperSettings = PrintPaperSettings(),
    ): NormalizedDoc

    suspend fun printSimplexDocument(
        printer: DiscoveredPrinter,
        doc: NormalizedDoc,
        jobName: String,
        paper: PrintPaperSettings = PrintPaperSettings(),
    ): PrintJobResult
}

/**
 * Discovery via SNMP + BJNP (official IJ LAN) and simplex print via BJNP/CLSS JPEG.
 */
class CanonSnmpSdkBridge(
    private val appContext: Context,
) : CanonSdkBridge {
    @Volatile
    private var nativeOk = false

    private val jpegRenderer = PdfJpegRenderer(appContext)
    private val imagePrep = ImageJpegPrep(appContext)
    private val printSession = ClssBjnpJpegSession()
    private val cloudConverter = CanonCloudDocConverter(appContext)

    override fun initNative(): InitNativeResult {
        if (nativeOk) {
            return InitNativeResult(true, "libsdk-core 已加载")
        }
        return try {
            System.loadLibrary("sdk-core")
            nativeOk = true
            InitNativeResult(true, "libsdk-core 已加载（SNMP+BJNP 发现 + CLSS 出纸）")
        } catch (t: Throwable) {
            nativeOk = false
            InitNativeResult(false, "native 加载失败: ${t.javaClass.simpleName}: ${t.message}")
        }
    }

    override suspend fun discoverPrinters(timeoutMs: Long): List<DiscoveredPrinter> =
        withContext(Dispatchers.IO) {
            val init = initNative()
            if (!init.ok) {
                return@withContext emptyList()
            }
            val broadcast = BroadcastAddress.resolve(appContext)
            IjParallelSearch(broadcast).search(timeoutMs.coerceAtLeast(3_000L))
        }

    override suspend fun printSimplexPdf(
        printer: DiscoveredPrinter,
        pdf: File,
        jobName: String,
        paper: PrintPaperSettings,
    ): PrintJobResult = withContext(Dispatchers.IO) {
        val init = initNative()
        if (!init.ok) {
            return@withContext PrintJobResult.Failed(init.message)
        }
        if (printer.ipAddress.isBlank() || printer.ipAddress == "0.0.0.0") {
            return@withContext PrintJobResult.Failed("无效打印机 IP")
        }
        try {
            val jpegs = jpegRenderer.renderAllPages(pdf)
            if (jpegs.isEmpty()) {
                return@withContext PrintJobResult.Failed("PDF 无页面可渲染")
            }
            printJpegsInternal(printer.ipAddress, jpegs, jobName, paper)
        } catch (t: Throwable) {
            PrintJobResult.Failed("出纸失败: ${t.message}", t)
        }
    }

    override suspend fun printJpegs(
        printer: DiscoveredPrinter,
        jpegs: List<File>,
        jobName: String,
        paper: PrintPaperSettings,
    ): PrintJobResult = withContext(Dispatchers.IO) {
        val init = initNative()
        if (!init.ok) {
            return@withContext PrintJobResult.Failed(init.message)
        }
        if (printer.ipAddress.isBlank() || printer.ipAddress == "0.0.0.0") {
            return@withContext PrintJobResult.Failed("无效打印机 IP")
        }
        if (jpegs.isEmpty()) {
            return@withContext PrintJobResult.Failed("无 JPEG 页可打印")
        }
        try {
            printJpegsInternal(printer.ipAddress, jpegs, jobName, paper)
        } catch (t: Throwable) {
            PrintJobResult.Failed("出纸失败: ${t.message}", t)
        }
    }

    override suspend fun normalizeToPrintable(
        uri: Uri,
        displayName: String?,
        paper: PrintPaperSettings,
        onCloudProgress: (CanonCloudDocConverter.Progress) -> Unit,
    ): NormalizedDoc = withContext(Dispatchers.IO) {
        val name = displayName ?: uri.lastPathSegment ?: "document"
        val lower = name.lowercase()
        val mime = appContext.contentResolver.getType(uri)?.lowercase().orEmpty()
        val isPdf = lower.endsWith(".pdf") || mime == "application/pdf"
        val isDoc = lower.endsWith(".doc") && !lower.endsWith(".docx") ||
            mime == "application/msword"
        val isDocx = lower.endsWith(".docx") ||
            mime.contains("wordprocessingml")

        when {
            isPdf -> {
                val dest = File(appContext.cacheDir, "picked_${System.currentTimeMillis()}.pdf")
                copyUri(uri, dest)
                NormalizedDoc.LocalPdf(dest)
            }
            isDoc || isDocx -> {
                if (!DocConvertAcceptance.isAccepted(appContext)) {
                    CloudLog.e("normalize", "Word 转换被拒绝：未接受云转换说明")
                    throw IllegalStateException("需要先接受佳能云文档转换说明")
                }
                val ext = if (isDocx) "docx" else "doc"
                val dest = File(appContext.cacheDir, "picked_${System.currentTimeMillis()}.$ext")
                try {
                    copyUri(uri, dest)
                } catch (t: Throwable) {
                    throw CloudConvertException(
                        "复制 Word 文件失败 uri=$uri: ${t.javaClass.simpleName}: ${t.message}",
                        t,
                        stage = "normalize",
                    )
                }
                CloudLog.i(
                    "normalize",
                    "Word 本地副本 ${dest.absolutePath} size=${dest.length()} mime=$mime paper=${paper.summary()}",
                )
                val outDir = File(appContext.cacheDir, "cloud_jpeg_${System.currentTimeMillis()}")
                outDir.mkdirs()
                try {
                    val pages = cloudConverter.convertToJpegPages(
                        file = dest,
                        contentType = ext,
                        downloadDir = outDir,
                        deviceName = "G3800",
                        printTicketXml = PrintTicketFactory.jpeg300("G3800", paper),
                        onProgress = onCloudProgress,
                    )
                    CloudLog.i("normalize", "Word→JPEG 完成 ${pages.size} 页 name=$name")
                    NormalizedDoc.JpegPages(pages, name)
                } catch (e: CloudConvertException) {
                    CloudLog.e(
                        "normalize",
                        "CloudConvertException stage=${e.stage}: ${e.message}",
                        e,
                    )
                    throw e
                } catch (t: Throwable) {
                    throw CloudConvertException(
                        "云端转换失败: ${t.javaClass.name}: ${t.message}",
                        t,
                        stage = "normalize",
                    )
                }
            }
            else -> {
                CloudLog.e("normalize", "不支持的文件类型 name=$name mime=$mime")
                throw IllegalArgumentException("不支持的文件类型: $name (mime=$mime)")
            }
        }
    }

    override suspend fun normalizeImages(
        uris: List<Uri>,
        paper: PrintPaperSettings,
    ): NormalizedDoc = withContext(Dispatchers.IO) {
        require(uris.isNotEmpty()) { "未选择图片" }
        val pages = imagePrep.urisToJpegPages(uris, paper)
        val label = if (uris.size == 1) "相册图片" else "相册图片 ×${uris.size}"
        CloudLog.i("normalize", "相册→JPEG 完成 ${pages.size} 页 paper=${paper.summary()}")
        NormalizedDoc.JpegPages(pages, label)
    }

    override suspend fun printSimplexDocument(
        printer: DiscoveredPrinter,
        doc: NormalizedDoc,
        jobName: String,
        paper: PrintPaperSettings,
    ): PrintJobResult = when (doc) {
        is NormalizedDoc.LocalPdf -> printSimplexPdf(printer, doc.pdf, jobName, paper)
        is NormalizedDoc.JpegPages -> printJpegs(printer, doc.pages, jobName, paper)
    }

    private fun printJpegsInternal(
        ip: String,
        jpegs: List<File>,
        jobName: String,
        paper: PrintPaperSettings,
    ): PrintJobResult {
        return when (val r = printSession.printJpegs(ip, jpegs, jobName, paper)) {
            is ClssBjnpJpegSession.Result.Success -> PrintJobResult.Success
            is ClssBjnpJpegSession.Result.Failed -> PrintJobResult.Failed(r.message, r.cause)
        }
    }

    private fun copyUri(uri: Uri, dest: File) {
        appContext.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(dest).use { output -> input.copyTo(output) }
        } ?: throw IllegalArgumentException("无法读取文件")
    }
}
