package com.example.g3800duplex.print

import com.example.g3800duplex.canon.BjnpLog
import h7.C1631a
import java.io.File
import java.util.Locale
import jp.co.canon.bsd.ad.sdk.core.clss.CLSSMakeCommand
import jp.co.canon.bsd.ad.sdk.core.clss.CLSSResponseCommon
import jp.co.canon.bsd.ad.sdk.core.clss.CLSS_Define
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSEndJobParam
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSPrintSettingsInfo
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSSendDataParam
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSSetConfigurationParam
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSStartJobParam
import k7.g
import k7.h

/**
 * Minimal CLSS simplex job over BJNP:8611.
 * Sequence: open(retry) → StartJob → SetConfiguration → (SendData + JPEG)* → EndJob.
 */
class ClssBjnpJpegSession {
    private val cmd = CLSSMakeCommand()

    fun printJpegs(
        ip: String,
        jpegFiles: List<File>,
        jobName: String = "G3800Duplex",
        paper: PrintPaperSettings = PrintPaperSettings(),
    ): Result {
        if (jpegFiles.isEmpty()) {
            return Result.Failed("no JPEG pages")
        }
        BjnpLog.i(
            "session",
            "printJpegs start ip=$ip pages=${jpegFiles.size} job=$jobName paper=${paper.summary()}",
        )
        val sock = C1631a()
        return try {
            val openRc = openWithRetry(sock, ip)
            if (openRc != 0) {
                val detail = BjnpLog.openRcMessage(openRc)
                BjnpLog.e("session", "BJNP open failed rc=$openRc ($detail) ip=$ip")
                return Result.Failed("BJNP open failed rc=$openRc ($detail) ip=$ip")
            }
            val jobId = startJob(sock, jobName) ?: run {
                BjnpLog.e("session", "StartJob failed / no jobID")
                return Result.Failed("StartJob failed / no jobID")
            }
            BjnpLog.i("session", "StartJob ok jobId=$jobId")
            if (!setConfiguration(sock, jobId, paper)) {
                endJobQuiet(sock, jobId)
                BjnpLog.e("session", "SetConfiguration failed jobId=$jobId paper=${paper.summary()}")
                return Result.Failed(
                    "SetConfiguration failed jobId=$jobId paper=${paper.summary()}",
                )
            }
            for ((index, jpeg) in jpegFiles.withIndex()) {
                val bytes = jpeg.readBytes()
                BjnpLog.d(
                    "session",
                    "page ${index + 1}/${jpegFiles.size} jpegBytes=${bytes.size} file=${jpeg.name}",
                )
                if (!sendDataHeader(sock, jobId, bytes.size)) {
                    endJobQuiet(sock, jobId)
                    return Result.Failed("SendData header failed page=${index + 1}")
                }
                if (!writeAll(sock, bytes)) {
                    endJobQuiet(sock, jobId)
                    return Result.Failed("JPEG payload write failed page=${index + 1}")
                }
                drain(sock, 2)
            }
            endJob(sock, jobId)
            BjnpLog.i("session", "printJpegs success jobId=$jobId pages=${jpegFiles.size}")
            Result.Success(jobId, jpegFiles.size, jobName)
        } catch (t: Throwable) {
            BjnpLog.e("session", "CLSS session error: ${t.message}", t)
            Result.Failed("CLSS session error: ${t.message}", t)
        } finally {
            try {
                sock.close()
            } catch (_: Throwable) {
            }
        }
    }

    /**
     * Probe BJNP session only (open + close). Used by 「测试连接」.
     */
    fun probe(ip: String, timeoutMs: Int = CLSS_Define.CLSS_APPLICATION_ID_CPIS_ANDROID): ProbeResult {
        val start = System.currentTimeMillis()
        BjnpLog.i("probe", "start ip=$ip timeoutMs=$timeoutMs")
        val sock = C1631a()
        return try {
            val rc = openWithRetry(sock, ip, timeoutMs)
            val ms = System.currentTimeMillis() - start
            if (rc == 0) {
                BjnpLog.i("probe", "OK ip=$ip ms=$ms")
                ProbeResult(true, "BJNP 会话建立成功（UDP 握手 + TCP:8611）", ms, rc)
            } else {
                val detail = BjnpLog.openRcMessage(rc)
                BjnpLog.e("probe", "FAIL rc=$rc ($detail) ip=$ip ms=$ms")
                ProbeResult(false, "BJNP 会话失败 rc=$rc ($detail)", ms, rc)
            }
        } catch (t: Throwable) {
            val ms = System.currentTimeMillis() - start
            BjnpLog.e("probe", "exception: ${t.message}", t)
            ProbeResult(
                false,
                "BJNP probe 异常: ${t.javaClass.simpleName}: ${t.message}",
                ms,
                -99,
                t,
            )
        } finally {
            try {
                sock.close()
            } catch (_: Throwable) {
            }
        }
    }

    /**
     * Official pattern: retry open while rc==-1 within timeout
     * (busy cookie / transient UDP miss). See printer.b resume path.
     */
    private fun openWithRetry(
        sock: C1631a,
        ip: String,
        timeoutMs: Int = CLSS_Define.CLSS_APPLICATION_ID_CPIS_ANDROID,
    ): Int {
        val window = g(timeoutMs)
        var attempt = 0
        var lastRc = -1
        while (true) {
            attempt++
            val t0 = System.currentTimeMillis()
            lastRc = sock.open(ip)
            val dt = System.currentTimeMillis() - t0
            BjnpLog.i(
                "openRetry",
                "attempt=$attempt rc=$lastRc (${BjnpLog.openRcMessage(lastRc)}) ms=$dt ip=$ip",
            )
            if (lastRc == 0) return 0
            // -1 = no reply / busy → retry on same socket instance (keeps cookie)
            // -2 / -3 = hard fail → stop
            if (lastRc != -1 || window.a()) {
                if (lastRc == -1 && window.a()) {
                    BjnpLog.e("openRetry", "timeout after ${timeoutMs}ms attempts=$attempt")
                }
                return lastRc
            }
            h.p(200)
        }
    }

    /**
     * Align with Canon PrinterCommunicator / PliConfigSender StartJob:
     * only serviceType / hostEnvID=0 / jobID / bidi. Do NOT set keyMisdetection=0
     * (default 65535 = unset) or HostEnvID=1 — G3000 then only replies GetStatus.
     */
    private fun startJob(sock: C1631a, jobName: String): Int? {
        val param = CLSSStartJobParam()
        param.setServiceType(0)
        param.setHostEnvID(0)
        param.setJobID(String.format(Locale.ENGLISH, "%1\$08d", 2))
        param.setBidi("1")
        // jobName is optional in IVEC; keep unset like official IJ StartJob path.
        BjnpLog.i(
            "StartJob",
            "params serviceType=0 hostEnvID=0 jobID=00000002 bidi=1 (jobName ignored for wire: $jobName)",
        )
        val xml = cmd.getStartJob(param) ?: run {
            BjnpLog.e("StartJob", "getStartJob returned null")
            return null
        }
        BjnpLog.d("StartJob", "xml=${xml.replace("\n", " ").take(400)}")
        val payload = h.f(xml) ?: run {
            BjnpLog.e("StartJob", "encode StartJob XML failed")
            return null
        }
        BjnpLog.d("StartJob", "xmlBytes=${payload.size} head=${BjnpLog.hex(payload, 24)}")
        if (!writeAll(sock, payload)) {
            BjnpLog.e("StartJob", "write failed")
            return null
        }
        // Official g.c(sock, 7, 0): match getOperationPair()==START_JOB (7).
        val text = awaitOperation(
            sock,
            expectPair = CLSS_Define.CLSS_OPERATION_START_JOB,
            expectServiceType = 0,
            stage = "StartJob",
            maxNullReads = 8,
            maxReads = 60,
        ) ?: run {
            try {
                sock.e()
            } catch (_: Throwable) {
            }
            BjnpLog.e("StartJob", "no StartJobResponse (operationPair=7)")
            return null
        }
        return try {
            val common = CLSSResponseCommon(text)
            BjnpLog.i(
                "StartJob",
                "got op=${common.operationID} pair=${common.operationPair} " +
                    "resp=${common.response} detail=${common.responseDetail} jobID=${common.jobID}",
            )
            if (common.response != CLSS_Define.CLSS_OPERATION_RESPONSE_OK) {
                BjnpLog.e(
                    "StartJob",
                    "rejected response=${common.response} detail=${common.responseDetail}",
                )
                null
            } else {
                common.jobID?.toIntOrNull()
            }
        } catch (t: Throwable) {
            BjnpLog.e("StartJob", "parse failed: ${t.message}", t)
            null
        }
    }

    private fun setConfiguration(
        sock: C1631a,
        jobId: Int,
        paper: PrintPaperSettings,
    ): Boolean {
        val settings = CLSSPrintSettingsInfo()
        settings.init()
        settings.papersize = paper.size.clssSize
        settings.mediatype = paper.media.clssMedia
        settings.borderlessprint = CLSS_Define.CLSS_IVEC_BORDERLESS_OFF
        settings.colormode = CLSS_Define.CLSS_IVEC_COLORMODE_COLOR
        settings.duplexprint = CLSS_Define.CLSS_IVEC_DUPLEX_OFF
        settings.quality = CLSS_Define.CLSS_PRINT_QUALITY_NORMAL

        val param = CLSSSetConfigurationParam()
        param.setJobID(String.format(Locale.ENGLISH, "%1\$08d", jobId))
        param.setServiceType(0)
        param.setPrintSettings(settings)

        val xml = cmd.getSetConfiguration(param, 1, "") ?: return false
        val payload = h.f(xml) ?: return false
        BjnpLog.d("SetConfig", "xmlBytes=${payload.size} paper=${paper.summary()}")
        if (!writeAll(sock, payload)) return false
        // Official waits for operationPair==SET_CONFIGURATION (5); tolerate miss and continue.
        val text = awaitOperation(
            sock,
            expectPair = CLSS_Define.CLSS_OPERATION_SET_CONFIGURATION,
            expectServiceType = 0,
            stage = "SetConfig",
            maxNullReads = 8,
            maxReads = 40,
        )
        if (text == null) {
            BjnpLog.w("SetConfig", "no SetConfigurationResponse; continuing")
            return true
        }
        return try {
            val common = CLSSResponseCommon(text)
            val ok = common.response == CLSS_Define.CLSS_OPERATION_RESPONSE_OK
            BjnpLog.i(
                "SetConfig",
                "op=${common.operationID} resp=${common.response} detail=${common.responseDetail} ok=$ok",
            )
            ok
        } catch (t: Throwable) {
            BjnpLog.w("SetConfig", "parse failed: ${t.message}")
            true
        }
    }

    /**
     * Official PrinterCommunicator.c: sleep + read until getOperationPair matches
     * (skip unrelated GetStatus etc.). [maxNullReads] empty reads abort; non-matching
     * payloads do not count against that budget.
     */
    private fun awaitOperation(
        sock: C1631a,
        expectPair: Int,
        expectServiceType: Int,
        stage: String,
        maxNullReads: Int,
        maxReads: Int,
    ): String? {
        var nullReads = 0
        var reads = 0
        while (reads < maxReads && nullReads < maxNullReads) {
            h.p(200)
            val resp = sock.read()
            if (resp == null) {
                nullReads++
                BjnpLog.d(stage, "await read=null nullReads=$nullReads/$maxNullReads")
                continue
            }
            reads++
            val text = String(resp, Charsets.UTF_8)
            try {
                val common = CLSSResponseCommon(text)
                val pair = try {
                    common.operationPair
                } catch (_: Throwable) {
                    -1
                }
                BjnpLog.d(
                    stage,
                    "await#$reads op=${common.operationID} pair=$pair " +
                        "svc=${common.serviceType} resp=${common.response} " +
                        "detail=${common.responseDetail} jobID=${common.jobID} " +
                        "snippet=${text.replace("\n", " ").take(160)}",
                )
                if (pair == expectPair && common.serviceType == expectServiceType) {
                    return text
                }
                // Also accept raw response op (pair+1) in case JNI pair map fails.
                if (common.operationID == expectPair + 1 &&
                    common.serviceType == expectServiceType
                ) {
                    return text
                }
            } catch (t: Throwable) {
                BjnpLog.w(stage, "await parse failed: ${t.message}")
            }
        }
        return null
    }

    private fun sendDataHeader(sock: C1631a, jobId: Int, jpegSize: Int): Boolean {
        val param = CLSSSendDataParam()
        param.setJobID(String.format(Locale.ENGLISH, "%1\$08d", jobId))
        param.setFormat(CLSS_Define.CLSS_FORMAT_JPEG)
        param.setDataSize(jpegSize.toLong())
        val xml = cmd.getSendData(param) ?: return false
        val payload = h.f(xml) ?: return false
        return writeAll(sock, payload)
    }

    private fun endJob(sock: C1631a, jobId: Int) {
        val param = CLSSEndJobParam()
        param.setJobID(String.format(Locale.ENGLISH, "%1\$08d", jobId))
        param.setServiceType(0)
        val xml = cmd.getEndJob(param) ?: return
        val payload = h.f(xml) ?: return
        writeAll(sock, payload)
        drain(sock, 8)
    }

    private fun endJobQuiet(sock: C1631a, jobId: Int) {
        try {
            endJob(sock, jobId)
        } catch (_: Throwable) {
        }
    }

    private fun writeAll(sock: C1631a, data: ByteArray): Boolean {
        var offset = 0
        var guard = 0
        while (offset < data.size && guard++ < 10_000) {
            val n = sock.write(data, offset, data.size - offset)
            if (n < 0) {
                h.p(200)
                continue
            }
            if (n == 0) {
                h.p(50)
                continue
            }
            offset += n
        }
        if (offset < data.size) {
            BjnpLog.e("write", "incomplete wrote=$offset/${data.size}")
        }
        return offset >= data.size
    }

    private fun drain(sock: C1631a, attempts: Int) {
        repeat(attempts) {
            h.p(150)
            sock.read()
        }
    }

    data class ProbeResult(
        val ok: Boolean,
        val message: String,
        val latencyMs: Long,
        val openRc: Int,
        val cause: Throwable? = null,
    )

    sealed class Result {
        data class Success(val jobId: Int, val pages: Int, val jobName: String) : Result()
        data class Failed(val message: String, val cause: Throwable? = null) : Result()
    }
}
