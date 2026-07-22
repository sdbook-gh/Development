package com.example.g3800duplex.transport

import com.example.g3800duplex.canon.DiscoveredPrinter
import com.example.g3800duplex.canon.PrintJobResult
import com.example.g3800duplex.print.PrintPaperSettings
import java.io.File

interface PrinterBackend {
    val protocol: PrintProtocol

    suspend fun prepare(): ConnectionResult

    suspend fun discover(timeoutMs: Long = 10_000): Pair<List<DiscoveredPrinter>, ConnectionResult>

    /** Connectivity check without committing a full document job when possible. */
    suspend fun probe(printer: DiscoveredPrinter): ConnectionResult

    suspend fun printTestPage(
        printer: DiscoveredPrinter,
        jobName: String = "test-page",
        paper: PrintPaperSettings = PrintPaperSettings(),
    ): PrintJobResult

    suspend fun printPdf(
        printer: DiscoveredPrinter,
        pdf: File,
        jobName: String,
        paper: PrintPaperSettings = PrintPaperSettings(),
    ): PrintJobResult
}
