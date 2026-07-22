package com.example.g3800duplex.cloud

import android.content.Context
import android.util.Base64
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import org.json.JSONObject
import kotlin.coroutines.coroutineContext

/**
 * Canon CNPS cloud document → JPEG pages (aligned with official UploadOperation flow).
 */
class CanonCloudDocConverter(
    private val appContext: Context,
) {
    data class Progress(
        val stage: String,
        val detail: String = "",
        val page: Int = 0,
        val totalPages: Int = 0,
    )

    suspend fun convertToJpegPages(
        file: File,
        contentType: String,
        downloadDir: File,
        deviceName: String = "G3800",
        printTicketXml: String = PrintTicketFactory.a4Jpeg300(deviceName),
        onProgress: (Progress) -> Unit = {},
    ): List<File> = withContext(Dispatchers.IO) {
        require(contentType == "doc" || contentType == "docx") {
            "unsupported contentType=$contentType"
        }
        CloudLog.i(
            "start",
            "Word→JPEG file=${file.name} size=${file.length()}B type=$contentType " +
                "device=$deviceName out=${downloadDir.absolutePath}",
        )
        downloadDir.mkdirs()
        onProgress(Progress("auth", "ATP 鉴权中…"))
        val session = try {
            CanonAtpAuth.ensureAccessToken(appContext)
        } catch (t: Throwable) {
            if (t is CloudConvertException) throw t
            throw CloudConvertException(
                "ATP 鉴权异常: ${t.javaClass.simpleName}: ${t.message}",
                t,
                stage = "auth",
            )
        }
        val bearer = Base64.encodeToString(session.accessToken.toByteArray(Charsets.UTF_8), Base64.NO_WRAP)
        val base = session.region.prtBaseUrl.trimEnd('/')
        val ua = session.userAgent
        CloudLog.i(
            "auth",
            "OK ps_code=${session.region.psCode} prt=$base tokenLen=${session.accessToken.length}",
        )

        var documentId: String? = null
        try {
            onProgress(Progress("upload", "创建转换任务…"))
            val created = createConvertJob(base, bearer, ua, contentType, file.name)
            documentId = created.documentId
            CloudLog.i(
                "upload",
                "created documentId=$documentId convertJobId=${created.convertJobId} " +
                    "documentUrl=${created.documentUrl}",
            )
            ensureActive()

            onProgress(Progress("upload", "上传文档…"))
            putBinary(created.putDocumentUrl, gzipFile(file), ua, "application/gzip", "putDocument")
            ensureActive()

            onProgress(Progress("upload", "上传 PrintTicket…"))
            putBinary(
                created.putTicketUrl,
                gzipBytes(printTicketXml.toByteArray(Charsets.UTF_8)),
                ua,
                "application/gzip",
                "putTicket",
            )
            ensureActive()

            onProgress(Progress("upload", "通知开始转换…"))
            // Official CONVERTNOTIFY PUTs to document.createdResourceUrl (mDocumentURL), not convertJob URL.
            notifyUploaded(created.documentUrl, bearer, ua, created.convertJobId)
            ensureActive()

            onProgress(Progress("poll", "等待云端转换…"))
            val deadline = System.currentTimeMillis() + MAX_WAIT_MS
            val ready = pollUntilReady(
                base = base,
                bearer = bearer,
                accessToken = session.accessToken,
                ua = ua,
                convertJobId = created.convertJobId,
                deadlineMs = deadline,
                onProgress = onProgress,
            )
            if (ready.totalPages >= 100) {
                throw CloudConvertException(
                    "页数过多（${ready.totalPages}≥100），官方同样拒绝",
                    stage = "poll",
                )
            }
            if (ready.totalPages <= 0) {
                throw CloudConvertException("转换结果无页面", stage = "poll")
            }
            CloudLog.i("poll", "ready pages=${ready.totalPages} dataUrls=${ready.dataUrls.size}")

            onProgress(Progress("download", "下载 JPEG 页…", 0, ready.totalPages))
            val pages = ArrayList<File>(ready.totalPages)
            for (i in 1..ready.totalPages) {
                ensureActive()
                // Official DownloadOperation: GET dataList URL with empty Authorization (no Bearer).
                // Bearer on /data/N triggers AWS API Gateway "Invalid key=value pair in Authorization".
                val fromList = ready.dataUrls.getOrNull(i - 1)
                val url = if (!fromList.isNullOrBlank()) {
                    fromList
                } else {
                    val tokenQ = URLEncoder.encode(session.accessToken, Charsets.UTF_8.name())
                    "$base/api/mob/1.0/convertjobs/${created.convertJobId}/data/$i?access_token=$tokenQ"
                }
                val jpeg = downloadPage(
                    url,
                    ua,
                    session.encryptionKey,
                    File(downloadDir, "cloud_page_%03d.jpg".format(i)),
                    page = i,
                )
                if (!isJpeg(jpeg)) {
                    throw CloudConvertException(
                        "第 $i 页不是有效 JPEG（缺少 FF D8） path=${jpeg.absolutePath} size=${jpeg.length()}",
                        stage = "download",
                    )
                }
                pages.add(jpeg)
                onProgress(Progress("download", "已下载 $i/${ready.totalPages}", i, ready.totalPages))
            }
            CloudLog.i("done", "Word→JPEG 成功 ${pages.size} 页")
            pages
        } catch (e: CancellationException) {
            CloudLog.w("cancel", "Word 转换被取消", e)
            throw e
        } catch (e: CloudConvertException) {
            throw e
        } catch (t: Throwable) {
            throw CloudConvertException(
                "Word 云转换未捕获异常: ${t.javaClass.name}: ${t.message}",
                t,
                stage = "convert",
            )
        } finally {
            documentId?.let { id ->
                try {
                    deleteDocument(base, bearer, ua, id)
                } catch (t: Throwable) {
                    CloudLog.w("cleanup", "删除云端文档失败 documentId=$id", t)
                }
            }
        }
    }

    private data class CreatedJob(
        val documentId: String,
        val documentUrl: String,
        val convertJobId: String,
        val convertJobUrl: String,
        val putDocumentUrl: String,
        val putTicketUrl: String,
    )

    private data class ReadyJob(
        val totalPages: Int,
        val dataUrls: List<String>,
    )

    private fun createConvertJob(
        base: String,
        bearer: String,
        ua: String,
        contentType: String,
        fileName: String,
    ): CreatedJob {
        // Official UploadOperation.postRequest: raw JSON body + Content-Type application/json
        // (NOT multipart — that path caused CNPS 10006 "Request is invalid").
        val json = buildString {
            append("{\r\n  \"document\": {\r\n  \"documentName\" : ")
            append(JSONObject.quote(fileName))
            append(",\r\n")
            append("    \"contentType\" : \"$contentType\",\r\n")
            append("    \"data\" : {\r\n      \"compressionType\" : \"gzip\"\r\n    }\r\n  },\r\n")
            append("  \"convertJob\": {\r\n    \"printTicket\" : {\r\n")
            append("      \"contentType\" : \"cjt-cpt\",\r\n")
            append("      \"compressionType\" : \"gzip\"\r\n    },\r\n")
            append("    \"output\" : {\r\n")
            append("      \"type\" : \"jpeg\",\r\n")
            append("      \"compressionType\" : \"gzip\"\r\n    }\r\n  }\r\n}\r\n")
        }
        val body = json.toByteArray(Charsets.UTF_8)
        val url = "$base/api/mob/1.0/documents/convert"
        val conn = open(url, "POST", ua).apply {
            setRequestProperty("Authorization", "Bearer $bearer")
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Cache-Control", "no-cache, no-store")
            doOutput = true
            setFixedLengthStreamingMode(body.size)
        }
        try {
            conn.outputStream.use { it.write(body) }
            val code = conn.responseCode
            val text = readBody(conn)
            CloudLog.http("createJob", "POST", url, code, text)
            if (code !in 200..299) {
                val hint = when {
                    code == 401 && text.contains("19001") ->
                        "（CNPS 鉴权失败；通常因 ATP 客户端身份不被允许，已改用官方 applicationId 重新注册）"
                    code == 401 -> "（CNPS Bearer 鉴权失败）"
                    else -> ""
                }
                throw CloudConvertException(
                    "创建转换任务失败 HTTP $code: ${text.take(800)}$hint",
                    stage = "createJob",
                )
            }
            try {
                val root = JSONObject(text)
                val document = root.getJSONObject("document")
                val convertJob = root.getJSONObject("convertJob")
                val documentId = document.getString("documentId")
                return CreatedJob(
                    documentId = documentId,
                    documentUrl = document.optString("createdResourceUrl").ifBlank {
                        "$base/api/mob/1.0/documents/$documentId"
                    },
                    convertJobId = convertJob.getString("convertJobId"),
                    convertJobUrl = convertJob.optString("createdResourceUrl").ifBlank {
                        "$base/api/mob/1.0/convertjobs/${convertJob.getString("convertJobId")}"
                    },
                    putDocumentUrl = document.getJSONObject("data").getString("url"),
                    putTicketUrl = convertJob.getJSONObject("printTicket").getString("url"),
                )
            } catch (t: Throwable) {
                throw CloudConvertException(
                    "解析创建任务响应失败: ${t.message} body=${text.take(500)}",
                    t,
                    stage = "createJob",
                )
            }
        } finally {
            conn.disconnect()
        }
    }

    private fun notifyUploaded(
        documentUrl: String,
        bearer: String,
        ua: String,
        convertJobId: String,
    ) {
        // Official: PUT document.createdResourceUrl + raw JSON (CustomUploadStream gzip=false).
        val json = buildString {
            append("\r\n{\r\n  \"data\" : {\r\n    \"uploaded\" : true\r\n  },\r\n")
            append("  \"convertJob\" : {\r\n")
            append("    \"convertJobId\" : ${JSONObject.quote(convertJobId)},\r\n")
            append("    \"printTicket\" : {\r\n      \"uploaded\" : true\r\n    }\r\n  }\r\n}\r\n")
        }.toByteArray(Charsets.UTF_8)
        val conn = open(documentUrl, "PUT", ua).apply {
            setRequestProperty("Authorization", "Bearer $bearer")
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Cache-Control", "no-cache, no-store")
            doOutput = true
            setFixedLengthStreamingMode(json.size)
        }
        try {
            conn.outputStream.use { it.write(json) }
            val code = conn.responseCode
            val text = readBody(conn)
            CloudLog.http("notify", "PUT", documentUrl, code, text)
            if (code !in 200..299) {
                throw CloudConvertException(
                    "转换通知失败 HTTP $code: ${text.take(800)}",
                    stage = "notify",
                )
            }
        } finally {
            conn.disconnect()
        }
    }

    private suspend fun pollUntilReady(
        base: String,
        bearer: String,
        accessToken: String,
        ua: String,
        convertJobId: String,
        deadlineMs: Long,
        onProgress: (Progress) -> Unit,
    ): ReadyJob {
        var waitSec = 2
        while (coroutineContext.isActive) {
            if (System.currentTimeMillis() > deadlineMs) {
                throw CloudConvertException(
                    "云端转换超时（约 ${MAX_WAIT_MS / 60_000} 分钟） convertJobId=$convertJobId",
                    stage = "poll",
                )
            }
            // Official requestIndexFileURL uses raw access_token query param.
            val tokenQ = URLEncoder.encode(accessToken, Charsets.UTF_8.name())
            val url = "$base/api/mob/1.0/convertjobs/$convertJobId?access_token=$tokenQ"
            val conn = open(url, "GET", ua).apply {
                setRequestProperty("Authorization", "Bearer $bearer")
                setRequestProperty("Cache-Control", "no-cache, no-store")
            }
            try {
                val code = conn.responseCode
                val text = readBody(conn)
                when (code) {
                    200, 201 -> {
                        CloudLog.i("poll", "HTTP $code body=${text.replace("\n", " ").take(500)}")
                        val root = JSONObject(text)
                        val status = root.optInt("status", -1)
                        val total = root.optInt("totalPages", 0)
                        val errMsg = root.optString("message")
                            .ifBlank { root.optString("errorMessage") }
                            .ifBlank { root.optString("error") }
                        onProgress(Progress("poll", "status=$status pages=$total", 0, total))
                        // Official: status 10/20 = still converting; 30/40 = ready.
                        if (total > 0 && status in READY_STATUSES) {
                            val urls = ArrayList<String>()
                            val list = root.optJSONArray("dataList")
                            if (list != null) {
                                for (i in 0 until list.length()) {
                                    val item = list.opt(i)
                                    val u = when (item) {
                                        is String -> item
                                        is JSONObject -> item.optString("url")
                                            .ifBlank { item.optString("createdResourceUrl") }
                                        else -> list.optString(i)
                                    }
                                    if (u.isNotBlank()) urls.add(u)
                                }
                            }
                            if (urls.isEmpty()) {
                                CloudLog.w(
                                    "poll",
                                    "ready but dataList empty/unparsed; will use /data/N?access_token=",
                                )
                            }
                            return ReadyJob(total, urls)
                        }
                        if (status in FAILED_STATUSES) {
                            throw CloudConvertException(
                                "云端转换失败 status=$status message=$errMsg body=${text.take(800)}",
                                stage = "poll",
                            )
                        }
                    }
                    401 -> {
                        CloudLog.http("poll", "GET", url.substringBefore('?'), code, text)
                        throw CloudConvertException(
                            "轮询鉴权失败 HTTP 401 body=${text.take(400)}",
                            stage = "poll",
                        )
                    }
                    else -> {
                        CloudLog.w(
                            "poll",
                            "非终态 HTTP $code body=${text.replace("\n", " ").take(400)}",
                        )
                    }
                }
            } finally {
                conn.disconnect()
            }
            kotlinx.coroutines.delay(waitSec * 1000L)
            waitSec = (waitSec * 2).coerceAtMost(30)
        }
        throw CancellationException("cancelled")
    }

    private fun downloadPage(
        url: String,
        ua: String,
        encryptionKey: String,
        outFile: File,
        page: Int,
    ): File {
        // Match official getAPOHttpLibrary("GET", /* authToken */ ""): no Authorization header.
        val conn = open(url, "GET", ua).apply {
            setRequestProperty("Accept-Encoding", "gzip")
            setRequestProperty("Cache-Control", "no-cache, no-store")
        }
        try {
            val code = conn.responseCode
            if (code !in 200..299) {
                val text = readBody(conn)
                CloudLog.http("download", "GET", url, code, text)
                throw CloudConvertException(
                    "下载第 $page 页失败 HTTP $code: ${text.take(500)}",
                    stage = "download",
                )
            }
            var raw = conn.inputStream.use { it.readBytes() }
            val encoding = conn.contentEncoding?.lowercase()
            if (encoding == "gzip" || (raw.size >= 2 && raw[0] == 0x1f.toByte() && raw[1] == 0x8b.toByte())) {
                raw = GZIPInputStream(raw.inputStream()).use { it.readBytes() }
            }
            val decrypted = Rc4Utility.decrypt(raw, encryptionKey)
            FileOutputStream(outFile).use { it.write(decrypted) }
            CloudLog.i(
                "download",
                "page=$page bytes=${decrypted.size} → ${outFile.name}",
            )
            return outFile
        } catch (e: CloudConvertException) {
            throw e
        } catch (t: Throwable) {
            throw CloudConvertException(
                "下载/解密第 $page 页异常: ${t.javaClass.simpleName}: ${t.message}",
                t,
                stage = "download",
            )
        } finally {
            conn.disconnect()
        }
    }

    private fun deleteDocument(base: String, bearer: String, ua: String, documentId: String) {
        val url = "$base/api/mob/1.0/documents/$documentId"
        val conn = open(url, "DELETE", ua).apply {
            setRequestProperty("Authorization", "Bearer $bearer")
            setRequestProperty("Cache-Control", "no-cache, no-store")
        }
        try {
            conn.responseCode
        } finally {
            conn.disconnect()
        }
    }

    private fun putBinary(
        url: String,
        body: ByteArray,
        ua: String,
        contentType: String,
        stage: String,
    ) {
        val conn = open(url, "PUT", ua).apply {
            setRequestProperty("Content-Type", contentType)
            setRequestProperty("Cache-Control", "no-cache, no-store")
            doOutput = true
            setFixedLengthStreamingMode(body.size)
        }
        try {
            conn.outputStream.use { it.write(body) }
            val code = conn.responseCode
            val text = if (code !in 200..299) readBody(conn) else ""
            CloudLog.http(stage, "PUT", url, code, text.ifBlank { "(${body.size} bytes uploaded)" })
            if (code !in 200..299) {
                throw CloudConvertException(
                    "上传失败[$stage] HTTP $code: ${text.take(800)}",
                    stage = stage,
                )
            }
        } catch (e: CloudConvertException) {
            throw e
        } catch (t: Throwable) {
            throw CloudConvertException(
                "上传异常[$stage]: ${t.javaClass.simpleName}: ${t.message}",
                t,
                stage = stage,
            )
        } finally {
            conn.disconnect()
        }
    }

    private fun open(url: String, method: String, ua: String): HttpURLConnection {
        return (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 10_000
            readTimeout = 10_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", ua)
        }
    }

    private fun readBody(conn: HttpURLConnection): String {
        val stream = try {
            if (conn.responseCode >= 400) conn.errorStream else conn.inputStream
        } catch (_: Exception) {
            conn.errorStream
        } ?: return ""
        return stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
    }

    private fun gzipFile(file: File): ByteArray {
        FileInputStream(file).use { input ->
            val bos = ByteArrayOutputStream()
            GZIPOutputStream(bos).use { gzip ->
                input.copyTo(gzip)
            }
            return bos.toByteArray()
        }
    }

    private fun gzipBytes(bytes: ByteArray): ByteArray {
        val bos = ByteArrayOutputStream()
        GZIPOutputStream(bos).use { it.write(bytes) }
        return bos.toByteArray()
    }

    private fun isJpeg(file: File): Boolean {
        FileInputStream(file).use { input ->
            val b0 = input.read()
            val b1 = input.read()
            return b0 == 0xFF && b1 == 0xD8
        }
    }

    companion object {
        private const val MAX_WAIT_MS = 1_200_000L // 20 min, official LIBRARY_TIMEOUT_TIME
        private val READY_STATUSES = setOf(30, 40)
        private val FAILED_STATUSES = setOf(50, 51)
    }
}
