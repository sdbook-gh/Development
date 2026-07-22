package com.example.g3800duplex.ui

import android.app.Activity
import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.g3800duplex.canon.CanonSdkBridge
import com.example.g3800duplex.canon.DiscoveredPrinter
import com.example.g3800duplex.canon.DiscoverySource
import com.example.g3800duplex.canon.NormalizedDoc
import com.example.g3800duplex.canon.PrintJobResult
import com.example.g3800duplex.cloud.CloudConvertException
import com.example.g3800duplex.cloud.CloudLog
import com.example.g3800duplex.cloud.DocConvertAcceptance
import com.example.g3800duplex.duplex.Binding
import com.example.g3800duplex.duplex.DuplexPhase
import com.example.g3800duplex.duplex.DuplexPrintController
import com.example.g3800duplex.duplex.DuplexState
import com.example.g3800duplex.duplex.PdfSplitter
import com.example.g3800duplex.print.DocPageSelection
import com.example.g3800duplex.print.PaperMedia
import com.example.g3800duplex.print.PaperSettingsStore
import com.example.g3800duplex.print.PaperSize
import com.example.g3800duplex.print.PrintPaperSettings
import com.example.g3800duplex.transport.ConnectionResult
import com.example.g3800duplex.transport.PrintProtocol
import com.example.g3800duplex.transport.PrinterBackendFactory
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.launch

private fun DiscoverySource.label(): String = when (this) {
    DiscoverySource.Snmp -> "SNMP"
    DiscoverySource.Bjnp -> "BJNP"
    DiscoverySource.Both -> "SNMP+BJNP"
}

private const val TAG = "G3800DuplexUI"

private val PICK_MIME_TYPES = arrayOf(
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
)

@Composable
fun DuplexPrintScreen(
    activity: Activity,
    bridge: CanonSdkBridge,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val splitter = remember { PdfSplitter(context) }
    val controller = remember { DuplexPrintController(splitter, bridge) }
    val paperStore = remember { PaperSettingsStore(context) }
    var paper by remember { mutableStateOf(paperStore.load()) }

    val protocol = PrintProtocol.CanonClss
    val backend = remember { PrinterBackendFactory.create(activity, bridge) }
    var connResult by remember { mutableStateOf<ConnectionResult?>(null) }

    var printers by remember { mutableStateOf<List<DiscoveredPrinter>>(emptyList()) }
    var selected by remember { mutableStateOf<DiscoveredPrinter?>(null) }
    var docName by remember { mutableStateOf<String?>(null) }
    var normalized by remember { mutableStateOf<NormalizedDoc?>(null) }
    var pendingUri by remember { mutableStateOf<Uri?>(null) }
    var pendingName by remember { mutableStateOf<String?>(null) }
    var binding by remember { mutableStateOf(Binding.LONG_EDGE) }
    var duplexState by remember { mutableStateOf(DuplexState()) }
    var showReload by remember { mutableStateOf(false) }
    var showTos by remember { mutableStateOf(false) }
    var converting by remember { mutableStateOf(false) }
    var reloadGate by remember { mutableStateOf<CompletableDeferred<Unit>?>(null) }
    var scanning by remember { mutableStateOf(false) }
    var probing by remember { mutableStateOf(false) }
    var testingPage by remember { mutableStateOf(false) }
    var paperSizeExpanded by remember { mutableStateOf(false) }
    var paperMediaExpanded by remember { mutableStateOf(false) }
    var pagesExpanded by remember { mutableStateOf(false) }
    var pageCount by remember { mutableIntStateOf(0) }
    var selectedPages by remember { mutableStateOf<Set<Int>>(emptySet()) }
    var pageRangeInput by remember { mutableStateOf("") }

    LaunchedEffect(Unit) {
        connResult = backend.prepare()
    }

    LaunchedEffect(normalized) {
        val doc = normalized
        if (doc == null) {
            pageCount = 0
            selectedPages = emptySet()
            pageRangeInput = ""
            return@LaunchedEffect
        }
        val n = DocPageSelection.pageCount(context, doc)
        pageCount = n
        selectedPages = if (n > 0) (1..n).toSet() else emptySet()
        pageRangeInput = if (n > 0) "1-$n" else ""
        pagesExpanded = n > 1
    }

    LaunchedEffect(duplexState) {
        if (duplexState.message.isNotBlank()) {
            Log.i(TAG, "[${duplexState.phase}] ${duplexState.message}")
        }
    }

    fun persistPaper(next: PrintPaperSettings) {
        paper = next
        paperStore.save(next)
    }

    fun docForPrint(): NormalizedDoc? {
        val doc = normalized ?: return null
        if (selectedPages.isEmpty()) return null
        if (pageCount > 0 && selectedPages.size == pageCount) return doc
        return DocPageSelection.subset(context, doc, selectedPages)
    }

    fun startNormalize(uri: Uri, name: String?) {
        converting = true
        duplexState = DuplexState(DuplexPhase.Idle, "准备文档…")
        scope.launch {
            try {
                val doc = bridge.normalizeToPrintable(uri, name, paper) { p ->
                    duplexState = DuplexState(
                        DuplexPhase.Idle,
                        "云端转换: ${p.detail.ifBlank { p.stage }}" +
                            if (p.totalPages > 0) " (${p.page}/${p.totalPages})" else "",
                    )
                }
                normalized = doc
                docName = when (doc) {
                    is NormalizedDoc.LocalPdf -> name ?: doc.pdf.name
                    is NormalizedDoc.JpegPages ->
                        "${doc.sourceName}（JPEG ${doc.pages.size} 页）"
                }
                duplexState = DuplexState(
                    DuplexPhase.Idle,
                    when (doc) {
                        is NormalizedDoc.LocalPdf -> "已选 PDF，可用私有协议出纸"
                        is NormalizedDoc.JpegPages ->
                            "已准备 ${doc.pages.size} 页 JPEG，可出纸"
                    },
                )
            } catch (t: Throwable) {
                normalized = null
                docName = null
                val stage = (t as? CloudConvertException)?.stage ?: "normalize"
                val chain = CloudLog.formatChain(t)
                CloudLog.e(
                    "ui",
                    "文档准备失败 stage=$stage name=$name chain=$chain",
                    t,
                )
                val detail = buildString {
                    append("文档准备失败")
                    if (t is CloudConvertException) append(" [$stage]")
                    append(": ")
                    append(t.message ?: t.javaClass.simpleName)
                    if (chain.isNotBlank() && chain != "${t.javaClass.simpleName}: ${t.message}") {
                        append("\n")
                        append(chain)
                    }
                    append("\n(logcat 过滤: ${CloudLog.TAG})")
                }
                duplexState = DuplexState(DuplexPhase.Failed, detail)
                connResult = ConnectionResult.fail(
                    protocol,
                    stage,
                    detail,
                    cause = t,
                )
            } finally {
                converting = false
                pendingUri = null
                pendingName = null
            }
        }
    }

    val pickDoc = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        val name = uri.lastPathSegment
        val lower = (name ?: "").lowercase()
        val isWord = lower.endsWith(".doc") || lower.endsWith(".docx") ||
            (context.contentResolver.getType(uri)?.contains("word") == true)
        if (isWord && !DocConvertAcceptance.isAccepted(context)) {
            pendingUri = uri
            pendingName = name
            showTos = true
        } else {
            startNormalize(uri, name)
        }
    }

    fun startNormalizeImages(uris: List<Uri>) {
        if (uris.isEmpty()) return
        converting = true
        duplexState = DuplexState(DuplexPhase.Idle, "正在准备图片…")
        scope.launch {
            try {
                val doc = bridge.normalizeImages(uris, paper)
                normalized = doc
                docName = when (doc) {
                    is NormalizedDoc.JpegPages ->
                        "${doc.sourceName}（JPEG ${doc.pages.size} 页）"
                    else -> "相册图片"
                }
                duplexState = DuplexState(
                    DuplexPhase.Idle,
                    "已选 ${uris.size} 张图片 → ${paper.summary()}，可单面/双面出纸",
                )
            } catch (t: Throwable) {
                normalized = null
                docName = null
                val detail = "图片准备失败: ${t.message ?: t.javaClass.simpleName}"
                duplexState = DuplexState(DuplexPhase.Failed, detail)
                connResult = ConnectionResult.fail(protocol, "album", detail, cause = t)
            } finally {
                converting = false
            }
        }
    }

    val pickImages = rememberLauncherForActivityResult(
        ActivityResultContracts.PickMultipleVisualMedia(maxItems = 20),
    ) { uris: List<Uri> ->
        if (uris.isEmpty()) return@rememberLauncherForActivityResult
        startNormalizeImages(uris)
    }

    if (showTos) {
        AlertDialog(
            onDismissRequest = {
                showTos = false
                pendingUri = null
                pendingName = null
            },
            title = { Text("佳能云文档转换") },
            text = {
                Text(
                    "Word（.doc/.docx）将上传至佳能 CNPS 云端转换为 JPEG 后再打印。" +
                        "本应用为研究自用，会走官方云接口，请勿上架分发。" +
                        "继续即表示你了解并接受该流程。",
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        DocConvertAcceptance.setAccepted(context, true)
                        showTos = false
                        val uri = pendingUri
                        val name = pendingName
                        if (uri != null) startNormalize(uri, name)
                    },
                ) { Text("接受并继续") }
            },
            dismissButton = {
                TextButton(
                    onClick = {
                        showTos = false
                        pendingUri = null
                        pendingName = null
                    },
                ) { Text("取消") }
            },
        )
    }

    if (showReload) {
        AlertDialog(
            onDismissRequest = {},
            title = { Text("请翻面装纸") },
            text = { Text(DuplexPrintController.RELOAD_HINT) },
            confirmButton = {
                TextButton(
                    onClick = {
                        showReload = false
                        reloadGate?.complete(Unit)
                    },
                ) { Text("已装好，打印背面") }
            },
        )
    }

    val busy = converting || scanning || probing || testingPage ||
        duplexState.phase in setOf(
            DuplexPhase.Splitting,
            DuplexPhase.PrintingFront,
            DuplexPhase.PrintingBack,
            DuplexPhase.WaitingReload,
        )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("G3800 私有协议打印", style = MaterialTheme.typography.headlineSmall)
        Text(
            "CLSS/BJNP 直连喷墨。用「测试连接」与「打印测试页」验证通路。",
            style = MaterialTheme.typography.bodySmall,
        )

        Text("连接协议", style = MaterialTheme.typography.titleMedium)
        Text(protocol.label, style = MaterialTheme.typography.bodyMedium)

        Text("纸张设置", style = MaterialTheme.typography.titleMedium)
        Text(
            "当前: ${paper.summary()}（测试页与出纸均按此设置）",
            style = MaterialTheme.typography.bodySmall,
        )

        CollapsibleSection(
            title = "纸张大小",
            summary = paper.size.label,
            expanded = paperSizeExpanded,
            onToggle = { paperSizeExpanded = !paperSizeExpanded },
            enabled = !busy,
        ) {
            PaperSize.entries.forEach { size ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(
                            selected = paper.size == size,
                            enabled = !busy,
                        ) { persistPaper(paper.copy(size = size)) }
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(
                        selected = paper.size == size,
                        onClick = { if (!busy) persistPaper(paper.copy(size = size)) },
                        enabled = !busy,
                    )
                    Text(size.label)
                }
            }
        }

        CollapsibleSection(
            title = "纸张类型",
            summary = paper.media.label,
            expanded = paperMediaExpanded,
            onToggle = { paperMediaExpanded = !paperMediaExpanded },
            enabled = !busy,
        ) {
            PaperMedia.entries.forEach { media ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(
                            selected = paper.media == media,
                            enabled = !busy,
                        ) { persistPaper(paper.copy(media = media)) }
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(
                        selected = paper.media == media,
                        onClick = { if (!busy) persistPaper(paper.copy(media = media)) },
                        enabled = !busy,
                    )
                    Text(media.label)
                }
            }
        }

        ConnectionResultBox(connResult)

        Button(
            onClick = {
                scope.launch {
                    scanning = true
                    duplexState = DuplexState(DuplexPhase.Idle, protocol.discoverHint)
                    val (list, result) = backend.discover()
                    printers = list
                    connResult = result
                    scanning = false
                    if (list.isNotEmpty()) {
                        selected = list.firstOrNull {
                            it.model.contains("G3800", ignoreCase = true) ||
                                it.model.contains("G3000", ignoreCase = true)
                        } ?: list.first()
                        duplexState = DuplexState(
                            DuplexPhase.Idle,
                            "发现 ${list.size} 项；已选 ${selected?.displayLine()}",
                        )
                    } else {
                        selected = null
                        duplexState = DuplexState(DuplexPhase.Idle, result.message)
                    }
                }
            },
            enabled = !busy,
            modifier = Modifier.fillMaxWidth(),
        ) { Text(if (scanning) protocol.discoverHint else "搜索打印机") }

        OutlinedButton(
            onClick = {
                val printer = selected
                scope.launch {
                    probing = true
                    connResult = if (printer == null) {
                        ConnectionResult.fail(
                            protocol,
                            "probe",
                            "请先搜索并选择打印机",
                        )
                    } else {
                        backend.probe(printer)
                    }
                    duplexState = DuplexState(
                        if (connResult?.ok == true) DuplexPhase.Idle else DuplexPhase.Failed,
                        connResult?.message ?: "",
                    )
                    probing = false
                }
            },
            enabled = !busy,
            modifier = Modifier.fillMaxWidth(),
        ) { Text(if (probing) "测试连接中…" else "测试连接") }

        OutlinedButton(
            onClick = {
                scope.launch {
                    val printer = selected
                    if (printer == null) {
                        connResult = ConnectionResult.fail(
                            protocol,
                            "test-page",
                            "请先搜索并选择打印机，再打印测试页",
                        )
                        return@launch
                    }
                    testingPage = true
                    duplexState = DuplexState(
                        DuplexPhase.PrintingFront,
                        "正在打印测试页（${protocol.label} · ${paper.summary()}）…",
                    )
                    when (val r = backend.printTestPage(printer, "g3800-test-page", paper)) {
                        is PrintJobResult.Success -> {
                            connResult = ConnectionResult.ok(
                                protocol,
                                "test-page",
                                "测试页任务已成功提交/完成（${paper.summary()}）",
                                endpoint = printer.endpointLine(),
                            )
                            duplexState = DuplexState(
                                DuplexPhase.Completed,
                                "测试页成功（${protocol.label} · ${paper.summary()}）",
                            )
                        }
                        is PrintJobResult.Failed -> {
                            connResult = ConnectionResult.fail(
                                protocol,
                                "test-page",
                                r.message,
                                endpoint = printer.endpointLine(),
                                cause = r.cause,
                            )
                            duplexState = DuplexState(
                                DuplexPhase.Failed,
                                "测试页失败: ${r.message}",
                            )
                        }
                    }
                    testingPage = false
                }
            },
            enabled = !busy,
            modifier = Modifier.fillMaxWidth(),
        ) { Text(if (testingPage) "测试页打印中…" else "打印测试页") }

        OutlinedButton(
            onClick = { pickDoc.launch(PICK_MIME_TYPES) },
            enabled = !busy,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("选择 PDF/Word")
        }

        OutlinedButton(
            onClick = {
                pickImages.launch(
                    PickVisualMediaRequest(
                        ActivityResultContracts.PickVisualMedia.ImageOnly,
                    ),
                )
            },
            enabled = !busy,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("选择图片（相册）")
        }

        if (scanning || converting || probing || testingPage) {
            CircularProgressIndicator(modifier = Modifier.align(Alignment.CenterHorizontally))
        }

        if (printers.isNotEmpty()) {
            Text("打印机", style = MaterialTheme.typography.titleMedium)
            printers.forEach { p ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(selected = selected == p) { selected = p }
                        .padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(selected = selected == p, onClick = { selected = p })
                    Column {
                        Text("${p.name} / ${p.model}")
                        Text(p.displayLine(), style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }

        Text("文档: ${docName ?: "未选择"}")

        if (normalized != null && pageCount > 0) {
            CollapsibleSection(
                title = "打印页码",
                summary = DocPageSelection.summary(selectedPages, pageCount),
                expanded = pagesExpanded,
                onToggle = { pagesExpanded = !pagesExpanded },
                enabled = !busy,
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    TextButton(
                        onClick = {
                            selectedPages = (1..pageCount).toSet()
                            pageRangeInput = "1-$pageCount"
                        },
                        enabled = !busy,
                    ) { Text("全选") }
                    TextButton(
                        onClick = {
                            selectedPages = emptySet()
                            pageRangeInput = ""
                        },
                        enabled = !busy,
                    ) { Text("清空") }
                    TextButton(
                        onClick = {
                            selectedPages = (1..pageCount).filter { it % 2 == 1 }.toSet()
                            pageRangeInput = DocPageSelection.formatRange(selectedPages)
                        },
                        enabled = !busy,
                    ) { Text("奇数") }
                    TextButton(
                        onClick = {
                            selectedPages = (1..pageCount).filter { it % 2 == 0 }.toSet()
                            pageRangeInput = DocPageSelection.formatRange(selectedPages)
                        },
                        enabled = !busy,
                    ) { Text("偶数") }
                }
                OutlinedTextField(
                    value = pageRangeInput,
                    onValueChange = { pageRangeInput = it },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !busy,
                    singleLine = true,
                    label = { Text("页码范围") },
                    placeholder = { Text("例如 1-3,5") },
                    supportingText = { Text("用逗号分隔，支持区间") },
                )
                Button(
                    onClick = {
                        selectedPages = DocPageSelection.parseRange(pageRangeInput, pageCount)
                    },
                    enabled = !busy && pageRangeInput.isNotBlank(),
                    modifier = Modifier.fillMaxWidth(),
                ) { Text("应用范围") }
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 220.dp)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(0.dp),
                ) {
                    (1..pageCount).chunked(4).forEach { rowPages ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.Start,
                        ) {
                            rowPages.forEach { page ->
                                val checked = page in selectedPages
                                Row(
                                    modifier = Modifier
                                        .weight(1f)
                                        .clickable(enabled = !busy) {
                                            selectedPages = if (checked) {
                                                selectedPages - page
                                            } else {
                                                selectedPages + page
                                            }
                                        }
                                        .padding(vertical = 2.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Checkbox(
                                        checked = checked,
                                        onCheckedChange = { on ->
                                            selectedPages = if (on) {
                                                selectedPages + page
                                            } else {
                                                selectedPages - page
                                            }
                                        },
                                        enabled = !busy,
                                    )
                                    Text("$page")
                                }
                            }
                            repeat(4 - rowPages.size) {
                                Row(modifier = Modifier.weight(1f)) {}
                            }
                        }
                    }
                }
            }
        }

        Text("装订边（手动双面）", style = MaterialTheme.typography.titleMedium)
        Row {
            listOf(Binding.LONG_EDGE to "长边", Binding.SHORT_EDGE to "短边").forEach { (b, label) ->
                Row(
                    modifier = Modifier
                        .selectable(selected = binding == b) { binding = b }
                        .padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(selected = binding == b, onClick = { binding = b })
                    Text(label)
                }
            }
        }

        Button(
            onClick = {
                val printer = selected
                val doc = try {
                    docForPrint()
                } catch (t: Throwable) {
                    duplexState = DuplexState(
                        DuplexPhase.Failed,
                        "页码选择无效: ${t.message ?: t.javaClass.simpleName}",
                    )
                    return@Button
                }
                if (printer == null || doc == null) return@Button
                val pageHint = DocPageSelection.summary(selectedPages, pageCount)
                scope.launch {
                    duplexState = DuplexState(
                        DuplexPhase.PrintingFront,
                        "正在单面 CLSS 出纸 → ${printer.model}@${printer.ipAddress}" +
                            "（${paper.summary()}；$pageHint）",
                    )
                    when (
                        val r = bridge.printSimplexDocument(
                            printer,
                            doc,
                            "g3800-simplex",
                            paper,
                        )
                    ) {
                        is PrintJobResult.Success -> {
                            duplexState = DuplexState(
                                DuplexPhase.Completed,
                                "单面出纸成功（CLSS JPEG → BJNP:8611；$pageHint）",
                            )
                            connResult = ConnectionResult.ok(
                                protocol,
                                "print",
                                "单面 CLSS 出纸成功",
                                endpoint = printer.ipAddress,
                            )
                        }
                        is PrintJobResult.Failed -> {
                            duplexState = DuplexState(
                                DuplexPhase.Failed,
                                "单面出纸失败: ${r.message}",
                            )
                            connResult = ConnectionResult.fail(
                                protocol,
                                "print",
                                r.message,
                                endpoint = printer.ipAddress,
                                cause = r.cause,
                            )
                        }
                    }
                }
            },
            enabled = normalized != null && selected != null &&
                selectedPages.isNotEmpty() && !busy,
            modifier = Modifier.fillMaxWidth(),
        ) { Text("单面 CLSS 出纸") }

        Button(
            onClick = {
                val printer = selected ?: return@Button
                val doc = try {
                    docForPrint()
                } catch (t: Throwable) {
                    duplexState = DuplexState(
                        DuplexPhase.Failed,
                        "页码选择无效: ${t.message ?: t.javaClass.simpleName}",
                    )
                    return@Button
                } ?: return@Button
                scope.launch {
                    val gate = CompletableDeferred<Unit>()
                    reloadGate = gate
                    controller.runDocument(
                        printer = printer,
                        doc = doc,
                        binding = binding,
                        onState = { st ->
                            duplexState = st
                            if (st.phase == DuplexPhase.WaitingReload) {
                                showReload = true
                            }
                            if (st.phase == DuplexPhase.Failed) {
                                connResult = ConnectionResult.fail(
                                    protocol,
                                    "duplex",
                                    st.message,
                                    endpoint = printer.ipAddress,
                                )
                            }
                        },
                        awaitReloadConfirmed = { gate.await() },
                        paper = paper,
                    )
                }
            },
            enabled = normalized != null && selected != null &&
                selectedPages.isNotEmpty() && !busy,
            modifier = Modifier.fillMaxWidth(),
        ) { Text("开始手动双面打印") }

        Text("状态: ${duplexState.phase}")
        Text(duplexState.message)
        if (duplexState.frontPages.isNotEmpty()) {
            Text("正面页: ${duplexState.frontPages}")
            Text("背面页: ${duplexState.backPages}")
        }
    }
}

@Composable
private fun CollapsibleSection(
    title: String,
    summary: String,
    expanded: Boolean,
    onToggle: () -> Unit,
    enabled: Boolean,
    content: @Composable () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable(enabled = enabled, onClick = onToggle)
                .padding(vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                "$title  ${if (expanded) "▾" else "▸"}",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(summary, style = MaterialTheme.typography.bodySmall)
        }
        if (expanded) {
            content()
        }
    }
}

@Composable
private fun ConnectionResultBox(result: ConnectionResult?) {
    val borderColor = when {
        result == null -> MaterialTheme.colorScheme.outline
        result.ok -> MaterialTheme.colorScheme.primary
        else -> MaterialTheme.colorScheme.error
    }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, borderColor)
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text("连接结果 / 错误", style = MaterialTheme.typography.titleMedium)
        if (result == null) {
            Text("尚未探测", style = MaterialTheme.typography.bodySmall)
        } else {
            Text(result.summaryLine(), style = MaterialTheme.typography.bodySmall)
            result.cause?.message?.let { causeMsg ->
                if (causeMsg.isNotBlank() && !result.message.contains(causeMsg)) {
                    Text(
                        "cause: ${result.cause.javaClass.simpleName}: $causeMsg",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
        }
    }
}

private fun DiscoveredPrinter.displayLine(): String {
    val parts = mutableListOf<String>()
    if (ipAddress.isNotBlank()) parts += ipAddress
    if (macAddress.isNotBlank()) parts += macAddress
    if (protocolLabel.isNotBlank()) parts += protocolLabel
    else parts += source.label()
    return parts.joinToString(" · ")
}

private fun DiscoveredPrinter.endpointLine(): String =
    when {
        ipAddress.isNotBlank() -> ipAddress
        else -> name
    }
