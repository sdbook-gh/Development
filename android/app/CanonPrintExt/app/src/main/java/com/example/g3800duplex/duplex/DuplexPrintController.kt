package com.example.g3800duplex.duplex

import com.example.g3800duplex.canon.CanonSdkBridge
import com.example.g3800duplex.canon.DiscoveredPrinter
import com.example.g3800duplex.canon.NormalizedDoc
import com.example.g3800duplex.canon.PrintJobResult
import com.example.g3800duplex.print.PrintPaperSettings
import java.io.File

enum class DuplexPhase {
    Idle,
    Splitting,
    PrintingFront,
    WaitingReload,
    PrintingBack,
    Completed,
    Failed,
}

data class DuplexState(
    val phase: DuplexPhase = DuplexPhase.Idle,
    val message: String = "",
    val frontPages: List<Int> = emptyList(),
    val backPages: List<Int> = emptyList(),
)

/**
 * Orchestrates manual duplex: simplex front job → reload → simplex back job.
 * Always sends duplex=OFF at the protocol layer (G3800 has no auto duplex unit).
 */
class DuplexPrintController(
    private val splitter: PdfSplitter,
    private val bridge: CanonSdkBridge,
) {
    suspend fun run(
        printer: DiscoveredPrinter,
        sourcePdf: File,
        binding: Binding,
        onState: (DuplexState) -> Unit,
        awaitReloadConfirmed: suspend () -> Unit,
        paper: PrintPaperSettings = PrintPaperSettings(),
    ): DuplexState {
        onState(DuplexState(DuplexPhase.Splitting, "正在拆分正反面 PDF…"))
        val split = try {
            splitter.split(sourcePdf, binding)
        } catch (t: Throwable) {
            return DuplexState(DuplexPhase.Failed, "拆分失败: ${t.message}").also(onState)
        }

        onState(
            DuplexState(
                DuplexPhase.PrintingFront,
                "正在打印正面 ${split.frontPageNumbers}",
                split.frontPageNumbers,
                split.backPageNumbers,
            ),
        )
        when (val front = bridge.printSimplexPdf(printer, split.frontPdf, "g3800-front", paper)) {
            is PrintJobResult.Failed -> {
                return DuplexState(
                    DuplexPhase.Failed,
                    "正面打印失败: ${front.message}",
                    split.frontPageNumbers,
                    split.backPageNumbers,
                ).also(onState)
            }
            PrintJobResult.Success -> Unit
        }

        if (split.backPdf == null) {
            return DuplexState(
                DuplexPhase.Completed,
                "仅奇数页，无需背面",
                split.frontPageNumbers,
                emptyList(),
            ).also(onState)
        }

        onState(
            DuplexState(
                DuplexPhase.WaitingReload,
                RELOAD_HINT,
                split.frontPageNumbers,
                split.backPageNumbers,
            ),
        )
        awaitReloadConfirmed()

        onState(
            DuplexState(
                DuplexPhase.PrintingBack,
                "正在打印背面 ${split.backPageNumbers}",
                split.frontPageNumbers,
                split.backPageNumbers,
            ),
        )
        when (val back = bridge.printSimplexPdf(printer, split.backPdf, "g3800-back", paper)) {
            is PrintJobResult.Failed -> {
                return DuplexState(
                    DuplexPhase.Failed,
                    "背面打印失败: ${back.message}",
                    split.frontPageNumbers,
                    split.backPageNumbers,
                ).also(onState)
            }
            PrintJobResult.Success -> Unit
        }

        return DuplexState(
            DuplexPhase.Completed,
            "手动双面完成",
            split.frontPageNumbers,
            split.backPageNumbers,
        ).also(onState)
    }

    /**
     * Manual duplex for ordered JPEG pages (e.g. Word cloud convert).
     * Uses the same odd/even [PageOrder] rules as PDF split.
     */
    suspend fun runJpegs(
        printer: DiscoveredPrinter,
        pages: List<File>,
        binding: Binding,
        onState: (DuplexState) -> Unit,
        awaitReloadConfirmed: suspend () -> Unit,
        paper: PrintPaperSettings = PrintPaperSettings(),
    ): DuplexState {
        onState(DuplexState(DuplexPhase.Splitting, "正在按页序拆分正反面 JPEG…"))
        if (pages.isEmpty()) {
            return DuplexState(DuplexPhase.Failed, "无 JPEG 页").also(onState)
        }
        val pageCount = pages.size
        val frontIdx = PageOrder.frontPages(pageCount) // 1-based
        val backIdx = PageOrder.backPages(pageCount, binding)
        val frontFiles = frontIdx.map { pages[it - 1] }
        val backFiles = backIdx.map { pages[it - 1] }

        onState(
            DuplexState(
                DuplexPhase.PrintingFront,
                "正在打印正面 $frontIdx",
                frontIdx,
                backIdx,
            ),
        )
        when (val front = bridge.printJpegs(printer, frontFiles, "g3800-front", paper)) {
            is PrintJobResult.Failed -> {
                return DuplexState(
                    DuplexPhase.Failed,
                    "正面打印失败: ${front.message}",
                    frontIdx,
                    backIdx,
                ).also(onState)
            }
            PrintJobResult.Success -> Unit
        }

        if (backFiles.isEmpty()) {
            return DuplexState(
                DuplexPhase.Completed,
                "仅奇数页，无需背面",
                frontIdx,
                emptyList(),
            ).also(onState)
        }

        onState(
            DuplexState(
                DuplexPhase.WaitingReload,
                RELOAD_HINT,
                frontIdx,
                backIdx,
            ),
        )
        awaitReloadConfirmed()

        onState(
            DuplexState(
                DuplexPhase.PrintingBack,
                "正在打印背面 $backIdx",
                frontIdx,
                backIdx,
            ),
        )
        when (val back = bridge.printJpegs(printer, backFiles, "g3800-back", paper)) {
            is PrintJobResult.Failed -> {
                return DuplexState(
                    DuplexPhase.Failed,
                    "背面打印失败: ${back.message}",
                    frontIdx,
                    backIdx,
                ).also(onState)
            }
            PrintJobResult.Success -> Unit
        }

        return DuplexState(
            DuplexPhase.Completed,
            "手动双面完成",
            frontIdx,
            backIdx,
        ).also(onState)
    }

    suspend fun runDocument(
        printer: DiscoveredPrinter,
        doc: NormalizedDoc,
        binding: Binding,
        onState: (DuplexState) -> Unit,
        awaitReloadConfirmed: suspend () -> Unit,
        paper: PrintPaperSettings = PrintPaperSettings(),
    ): DuplexState = when (doc) {
        is NormalizedDoc.LocalPdf ->
            run(printer, doc.pdf, binding, onState, awaitReloadConfirmed, paper)
        is NormalizedDoc.JpegPages ->
            runJpegs(printer, doc.pages, binding, onState, awaitReloadConfirmed, paper)
    }

    companion object {
        const val RELOAD_HINT =
            "正面已发送。请取出已打印的纸张，按进纸方向翻面后重新放入纸盒（空白面朝向打印头一侧），确认无误后继续打印背面。"
    }
}
