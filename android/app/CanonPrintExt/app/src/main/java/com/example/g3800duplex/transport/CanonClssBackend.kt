package com.example.g3800duplex.transport

import android.content.Context
import com.example.g3800duplex.canon.BjnpLog
import com.example.g3800duplex.canon.CanonSdkBridge
import com.example.g3800duplex.canon.DiscoveredPrinter
import com.example.g3800duplex.canon.PrintJobResult
import com.example.g3800duplex.print.ClssBjnpJpegSession
import com.example.g3800duplex.print.PrintPaperSettings
import com.example.g3800duplex.print.TestPageGenerator
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class CanonClssBackend(
    private val context: Context,
    private val bridge: CanonSdkBridge,
) : PrinterBackend {
    override val protocol: PrintProtocol = PrintProtocol.CanonClss
    private val testPages = TestPageGenerator(context)

    override suspend fun prepare(): ConnectionResult = withContext(Dispatchers.IO) {
        val start = System.currentTimeMillis()
        val init = bridge.initNative()
        val ms = System.currentTimeMillis() - start
        if (init.ok) {
            ConnectionResult.ok(protocol, "prepare", init.message, latencyMs = ms)
        } else {
            ConnectionResult.fail(protocol, "prepare", init.message, latencyMs = ms)
        }
    }

    override suspend fun discover(timeoutMs: Long): Pair<List<DiscoveredPrinter>, ConnectionResult> =
        withContext(Dispatchers.IO) {
            val start = System.currentTimeMillis()
            val prep = bridge.initNative()
            if (!prep.ok) {
                return@withContext emptyList<DiscoveredPrinter>() to ConnectionResult.fail(
                    protocol,
                    "discover",
                    "发现前 native 未就绪: ${prep.message}",
                )
            }
            val list = try {
                bridge.discoverPrinters(timeoutMs).map {
                    it.copy(protocolLabel = protocol.label)
                }
            } catch (t: Throwable) {
                return@withContext emptyList<DiscoveredPrinter>() to ConnectionResult.fail(
                    protocol,
                    "discover",
                    "SNMP+BJNP 搜索异常: ${t.javaClass.simpleName}: ${t.message}",
                    cause = t,
                    latencyMs = System.currentTimeMillis() - start,
                )
            }
            val ms = System.currentTimeMillis() - start
            if (list.isEmpty()) {
                emptyList<DiscoveredPrinter>() to ConnectionResult.fail(
                    protocol,
                    "discover",
                    "未发现喷墨打印机。请确认与打印机同一 Wi‑Fi，允许局域网/位置权限与组播后重试。",
                    latencyMs = ms,
                )
            } else {
                list to ConnectionResult.ok(
                    protocol,
                    "discover",
                    "发现 ${list.size} 台（SNMP+BJNP）",
                    endpoint = list.first().ipAddress,
                    latencyMs = ms,
                )
            }
        }

    override suspend fun probe(printer: DiscoveredPrinter): ConnectionResult =
        withContext(Dispatchers.IO) {
            val init = bridge.initNative()
            if (!init.ok) {
                return@withContext ConnectionResult.fail(
                    protocol,
                    "probe",
                    "native 未就绪: ${init.message}",
                    endpoint = printer.ipAddress,
                )
            }
            if (printer.ipAddress.isBlank() || printer.ipAddress == "0.0.0.0") {
                return@withContext ConnectionResult.fail(
                    protocol,
                    "probe",
                    "无效打印机 IP",
                    endpoint = printer.ipAddress,
                )
            }
            val endpoint = "${printer.ipAddress}:8611"
            // Bare TCP:8611 often ECONNREFUSED until UDP session-open — that is normal for BJNP.
            val tcp = TcpProbe.connect(printer.ipAddress, 8611, 1_500)
            BjnpLog.i(
                "probe",
                "TCP precheck (informational) ${if (tcp.ok) "OK" else "expected-refuse?"}: ${tcp.message}",
            )
            val bjnp = ClssBjnpJpegSession().probe(printer.ipAddress)
            val msg = if (bjnp.ok) {
                buildString {
                    append(bjnp.message)
                    if (!tcp.ok) {
                        append("；裸 TCP 预检拒绝属正常（需先 UDP 握手）")
                    }
                }
            } else {
                buildString {
                    append(bjnp.message)
                    append("；TCP预检: ")
                    append(tcp.message)
                    if (bjnp.openRc == -1 && tcp.ok) {
                        append("（TCP 通但 UDP 握手失败）")
                    }
                    if (bjnp.openRc == -3) {
                        append("（UDP 已拿到 session，但 TCP:8611 失败）")
                    }
                }
            }
            if (bjnp.ok) {
                ConnectionResult.ok(
                    protocol,
                    "probe",
                    msg,
                    endpoint = endpoint,
                    latencyMs = bjnp.latencyMs,
                )
            } else {
                ConnectionResult.fail(
                    protocol,
                    "probe",
                    msg,
                    endpoint = endpoint,
                    latencyMs = bjnp.latencyMs,
                    cause = bjnp.cause,
                )
            }
        }

    override suspend fun printTestPage(
        printer: DiscoveredPrinter,
        jobName: String,
        paper: PrintPaperSettings,
    ): PrintJobResult =
        withContext(Dispatchers.IO) {
            val pdf = testPages.generate(
                protocol,
                "${printer.ipAddress}:8611",
                printer.model,
                paper,
            )
            bridge.printSimplexPdf(printer, pdf, jobName, paper)
        }

    override suspend fun printPdf(
        printer: DiscoveredPrinter,
        pdf: File,
        jobName: String,
        paper: PrintPaperSettings,
    ): PrintJobResult = bridge.printSimplexPdf(printer, pdf, jobName, paper)
}
