"""Render pasted Word content as HTML in Chromium (QWebEngineView).

Word COM calls run in a worker thread (pythoncom.CoInitialize) so a slow
Word startup or a VBE dialog cannot freeze the Qt UI.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Callable, Optional

import win32clipboard
import win32com.client
import win32con
import win32gui
from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

from app_logging import get_logger
from word_host.html_to_rtf import html_to_rtf

log = get_logger("word_host.html")

_MSO_AUTOMATION_SECURITY_FORCE_DISABLE = 3
_WD_FORMAT_FILTERED_HTML = 10

_BLANK_HTML = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<style>body{font-family:'Microsoft YaHei',sans-serif;font-size:11pt;"
    "line-height:1.5;padding:8px;} img{max-width:100%;}</style></head>"
    "<body contenteditable='true' id='root'></body></html>"
)


class _WordWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[], object]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        import pythoncom

        pythoncom.CoInitialize()
        try:
            result = self._fn()
            self.finished.emit(result)
        except Exception as exc:
            log.exception("WordWorker failed")
            self.failed.emit(str(exc))
        finally:
            pythoncom.CoUninitialize()


class _Relay(QObject):
    """Lives in the main thread; relays worker signals so slots run on main."""

    done = Signal(object)
    failed = Signal(str)


class _PasteView(QWebEngineView):
    """QWebEngineView that intercepts Ctrl+V to capture Word's raw CF_HTML."""

    def __init__(self, host: "WordHtmlHost", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._host = host

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.matches(QKeySequence.StandardKey.Paste):
            if self._host._handle_paste():
                event.accept()
                return
        super().keyPressEvent(event)


class WordHtmlHost:
    """HTML/Chromium editing surface with Word-compatible save."""

    def __init__(self, container: QWidget) -> None:
        self.container = container
        self.container.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._view = _PasteView(self, container)
        self._view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self._view)

        self._path: Optional[str] = None
        self._loaded: bool = False
        self._workers: list[tuple[QThread, _WordWorker]] = []
        # Raw Word CF_HTML captured at paste time; used for saving so the
        # exported file is based on the same source as the rendered view,
        # instead of Chromium's lossy toHtml() serialization.
        self._paste_html: Optional[str] = None
        log.debug("WordHtmlHost created")

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def create_blank(self) -> None:
        log.info("create_blank: start")
        self._path = None
        self._paste_html = None
        self._loaded = True
        self._view.setHtml(_BLANK_HTML, QUrl("about:blank"))
        log.info("create_blank: done")

    def load(
        self,
        path: str,
        on_done: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        abs_path = os.path.abspath(path)
        log.info("load: start path=%s", abs_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(abs_path)

        self._path = abs_path
        self._paste_html = None
        self._loaded = True

        def _work() -> str:
            html_path = self._doc_to_html(abs_path)
            if not html_path or not os.path.isfile(html_path):
                raise RuntimeError("无法将 Word 文件转换为 HTML 以供渲染。")
            return html_path

        def _done(html_path: object) -> None:
            self._view.setUrl(QUrl.fromLocalFile(str(html_path)))
            log.info("load: rendered from %s", html_path)
            if on_done:
                on_done(str(html_path))

        def _fail(msg: str) -> None:
            if on_error:
                on_error(msg)

        self._run_in_thread(_work, _done, _fail)

    def save_as(self, path: str) -> str:
        """Save the editor HTML to .docx (html-for-docx) or .rtf (html_to_rtf).

        Pure Python, no Word COM. Before converting, Word's mso-* inline styles
        (which html4docx skips) are translated to standard CSS that html4docx
        understands, so fonts/sizes/colors/indent/shading are preserved.
        """
        log.info("save_as: request path=%s", path)
        if not self._loaded:
            raise RuntimeError("没有可保存的内容，请先创建编辑区或打开文件。")

        abs_path = os.path.abspath(path)
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in (".docx", ".rtf"):
            abs_path = abs_path + ".docx"
            ext = ".docx"

        # Prefer the raw Word CF_HTML captured at paste time: it carries Word's
        # full inline styling. Falls back to toHtml() for typed content.
        html = self._paste_html
        if not html:
            html = self._read_html_sync()
        if not html:
            raise RuntimeError("无法读取当前编辑区 HTML。")

        # Translate mso-* styles to standard CSS, then drop transient file://
        # images that html4docx/pandoc cannot fetch.
        html = self._preprocess_mso_styles(html)
        html = self._sanitize_images(html)

        if ext == ".rtf":
            rtf = html_to_rtf(html)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(rtf)
            log.info("save_as: rtf ok -> %s (%d bytes)", abs_path, len(rtf))
        else:
            # .docx：交由系统安装的 Microsoft Word 打开 HTML 并另存为 .docx
            if not self._word_available():
                raise RuntimeError(
                    "未检测到 Microsoft Word，无法将内容保存为 .docx 文件。\n"
                    "请确认已安装 Microsoft Word，或改为保存为 .rtf 格式。"
                )
            self._run_word_sync(lambda: self._save_docx_via_word(html, abs_path))
            log.info("save_as: docx ok -> %s", abs_path)

        self._path = abs_path
        return abs_path

    @staticmethod
    def _word_available() -> bool:
        """检测系统是否安装了可用于自动化保存的 Microsoft Word。

        通过尝试创建 Word COM 对象来判断；成功则立即退出该实例。
        返回 True 表示可用，False 表示未安装或 COM 不可用。
        """
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        word = None
        try:
            word = win32com.client.DispatchEx("Word.Application")
            return True
        except Exception:
            log.debug("Word availability check failed", exc_info=True)
            return False
        finally:
            if word is not None:
                try:
                    word.Quit()
                except Exception:
                    pass
            pythoncom.CoUninitialize()

    def _save_docx_via_word(self, html: str, abs_path: str) -> None:
        """使用系统 Word 把 HTML 内容保存为 .docx。

        流程：把 HTML 写入临时文件 -> Word 打开 -> SaveAs2(docx) -> 清理。
        要求本机已安装 Word（调用前应先用 _word_available 检测）。
        """
        import pythoncom
        import shutil
        import tempfile
        import win32com.client

        tmp_dir = tempfile.mkdtemp(prefix="word_app_save_")
        html_path = os.path.join(tmp_dir, "content.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        pythoncom.CoInitialize()
        word = None
        doc = None
        try:
            log.info("_save_docx_via_word: DispatchEx Word.Application")
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0
            try:
                word.AutomationSecurity = _MSO_AUTOMATION_SECURITY_FORCE_DISABLE
            except Exception:
                pass

            doc = word.Documents.Open(
                FileName=html_path,
                ConfirmConversions=False,
                ReadOnly=True,
                AddToRecentFiles=False,
            )
            # 16 = wdFormatDocumentDefault (.docx)
            doc.SaveAs2(FileName=abs_path, FileFormat=16)
            log.info("_save_docx_via_word: saved %s", abs_path)
        finally:
            if doc is not None:
                try:
                    doc.Close(SaveChanges=False)
                except Exception:
                    pass
            if word is not None:
                try:
                    word.Quit()
                except Exception:
                    pass
            pythoncom.CoUninitialize()
            self._hide_leftover_word_windows()
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def _run_word_sync(self, work: Callable[[], object], timeout_s: float = 45.0) -> object:
        """在后台线程运行 Word COM 任务并同步等待结果。

        复用 _run_in_thread 的看门狗/超时/taskkill 保护，避免 Word 卡死
        冻结 UI；通过 QEventLoop 维持同步调用契约（save_as 仍是同步返回）。
        """
        from PySide6.QtCore import QEventLoop

        state: dict[str, object] = {"value": None, "error": None}
        loop = QEventLoop()

        def _done(result: object) -> None:
            state["value"] = result
            loop.quit()

        def _fail(msg: str) -> None:
            state["error"] = msg
            loop.quit()

        self._run_in_thread(work, _done, _fail, timeout_s=timeout_s)

        guard = QTimer(self.container)
        guard.setSingleShot(True)

        def _on_guard() -> None:
            if state["error"] is None:
                state["error"] = f"Word 操作超时（{int(timeout_s)}s）。"
            loop.quit()

        guard.timeout.connect(_on_guard)
        guard.start(int((timeout_s + 5) * 1000))
        loop.exec()
        guard.stop()

        if state["error"] is not None:
            raise RuntimeError(str(state["error"]))
        return state["value"]

    @staticmethod
    def _preprocess_mso_styles(html: str) -> str:
        """Translate Word's mso-* inline styles to standard CSS.

        html4docx only reads standard CSS (font-size, font-family, color,
        background-color, text-indent, line-height, margin-left/right,
        text-align). Word's CF_HTML stores the real values in mso-* props:
        - mso-ansi-font-size / mso-bidi-font-size -> font-size
        - mso-fareast-font-family / mso-bidi-font-family -> font-family
        - mso-color-alt -> color
        - background -> background-color
        - mso-char-indent-count -> text-indent (N chars * base size)
        - mso-line-height-alt -> line-height
        - mso-margin-left-alt -> margin-left
        Standard props already present are kept (mso-* only fills gaps).
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for el in soup.find_all(attrs={"style": True}):
            raw = el["style"].strip().rstrip(";")
            props: dict[str, str] = {}
            for part in raw.split(";"):
                if ":" not in part:
                    continue
                k, v = part.split(":", 1)
                props[k.strip().lower()] = v.strip()

            add: list[tuple[str, str]] = []

            def _need(*keys: str) -> bool:
                return all(k not in props for k in keys)

            # font-size
            if _need("font-size"):
                for k in ("mso-ansi-font-size", "mso-bidi-font-size"):
                    if k in props:
                        add.append(("font-size", props[k]))
                        break
            # font-family (CJK font lives in fareast)
            if _need("font-family"):
                for k in ("mso-fareast-font-family", "mso-font-family", "mso-bidi-font-family"):
                    if k in props:
                        add.append(("font-family", props[k]))
                        break
            # color
            if _need("color") and "mso-color-alt" in props:
                add.append(("color", props["mso-color-alt"]))
            # background -> background-color
            if _need("background-color") and "background" in props:
                add.append(("background-color", props["background"]))
            # text-indent: mso-char-indent-count is in characters; Word default
            # CJK body size is 10.5pt (五号), so indent = count * 10.5pt.
            if _need("text-indent") and "mso-char-indent-count" in props:
                try:
                    n = float(props["mso-char-indent-count"])
                    add.append(("text-indent", f"{n * 10.5:.2f}pt"))
                except ValueError:
                    pass
            # line-height
            if _need("line-height") and "mso-line-height-alt" in props:
                add.append(("line-height", props["mso-line-height-alt"]))
            # margin-left
            if _need("margin-left") and "mso-margin-left-alt" in props:
                add.append(("margin-left", props["mso-margin-left-alt"]))

            if add:
                extra = "".join(f";{k}:{v}" for k, v in add)
                el["style"] = raw + extra
        return str(soup)

    def close(self) -> None:
        log.info("close: html host")
        self._path = None
        self._paste_html = None
        self._loaded = False
        try:
            self._view.setHtml("")
        except Exception:
            log.debug("close: clear view failed", exc_info=True)

    @staticmethod
    def _sanitize_images(html: str) -> str:
        """Remove <img> tags whose src is not an inline data URI.

        Word's clipboard HTML emits <img src="file:///.../clip_imageNNN.*">
        pointing at temp files that are gone by save time, which makes
        html4docx raise OSError. Inline data-URI images are preserved.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        removed = 0
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if src.startswith("data:"):
                continue
            alt = img.get("alt", "")
            if alt:
                img.replace_with(alt)
            else:
                img.decompose()
            removed += 1
        if removed:
            log.info("sanitize_images: removed %d non-inline img(s)", removed)
        return str(soup)

    def _handle_paste(self) -> bool:
        """Intercept Ctrl+V: grab Word's raw CF_HTML, insert it, and store it.

        Returns True if the paste was handled (CF_HTML found), False to let
        QWebEngineView fall back to its default paste handling.
        """
        cf_html = self._read_clipboard_cf_html()
        if not cf_html:
            return False
        fragment = self._extract_fragment(cf_html)
        if not fragment:
            return False
        self._paste_html = fragment
        log.info("paste: captured CF_HTML (%d chars)", len(fragment))
        # Insert the Word fragment at the caret so the rendered view and the
        # saved file share one HTML source.
        js = (
            "document.execCommand('insertHTML', false, "
            + json.dumps(fragment)
            + ");"
        )
        self._view.page().runJavaScript(js)
        return True

    @staticmethod
    def _read_clipboard_cf_html() -> Optional[bytes]:
        """Read the raw 'HTML Format' clipboard payload, or None."""
        try:
            cf = win32clipboard.RegisterClipboardFormat("HTML Format")
        except Exception:
            return None
        try:
            win32clipboard.OpenClipboard()
            try:
                if not win32clipboard.IsClipboardFormatAvailable(cf):
                    return None
                return win32clipboard.GetClipboardData(cf)
            finally:
                win32clipboard.CloseClipboard()
        except Exception:
            log.debug("read clipboard CF_HTML failed", exc_info=True)
            return None

    @staticmethod
    def _extract_fragment(cf_html: bytes) -> str:
        """Extract the fragment HTML between StartFragment/EndFragment offsets."""
        try:
            text = cf_html.decode("utf-8", errors="replace")
        except Exception:
            return ""
        # CF_HTML header carries byte offsets: StartFragment:NNN EndFragment:NNN
        m = re.search(r"StartFragment:(\d+)\s+EndFragment:(\d+)", text)
        if m:
            s, e = int(m.group(1)), int(m.group(2))
            # Offsets are byte-based; decode the slice from the raw bytes.
            try:
                return cf_html[s:e].decode("utf-8", errors="replace")
            except Exception:
                return text
        # Fallback: strip the header and return everything after the first blank line.
        idx = text.find("<html")
        return text[idx:] if idx >= 0 else text


    def _run_in_thread(
        self,
        work: Callable[[], object],
        on_done: Callable[[object], None],
        on_fail: Callable[[str], None],
        timeout_s: float = 45.0,
    ) -> None:
        thread = QThread()
        worker = _WordWorker(work)
        worker.moveToThread(thread)
        # Relay lives in the main thread so the lambdas below execute there,
        # not in the worker thread (avoids cross-thread QObject/timer errors and
        # unsafe UI updates).
        relay = _Relay()
        state = {"done": False}

        watchdog = QTimer(self.container)
        watchdog.setSingleShot(True)

        def _finish(result: object) -> None:
            if state["done"]:
                return
            state["done"] = True
            watchdog.stop()
            thread.quit()
            on_done(result)

        def _fail(msg: str) -> None:
            if state["done"]:
                return
            state["done"] = True
            watchdog.stop()
            thread.quit()
            on_fail(msg)

        def _timeout() -> None:
            if state["done"]:
                return
            state["done"] = True
            log.warning("Word COM worker timed out after %ss; killing WINWORD", timeout_s)
            self._kill_winword()
            thread.quit()
            on_fail(
                f"Word 操作超时（{int(timeout_s)}s）：Word 可能弹出了对话框（如 VBA 环境）。"
                "已尝试终止 Word 进程，请重试。"
            )

        watchdog.timeout.connect(_timeout)
        watchdog.start(int(timeout_s * 1000))

        def _cleanup() -> None:
            watchdog.stop()
            watchdog.deleteLater()
            relay.deleteLater()
            try:
                self._workers.remove((thread, worker))
            except ValueError:
                pass
            worker.deleteLater()
            thread.deleteLater()

        thread.started.connect(worker.run)
        # Worker (worker thread) -> relay (main thread): auto = queued.
        worker.finished.connect(relay.done)
        worker.failed.connect(relay.failed)
        # Relay (main thread) -> lambdas: direct, executes on the main thread.
        relay.done.connect(_finish)
        relay.failed.connect(_fail)
        thread.finished.connect(_cleanup)

        self._workers.append((thread, worker))
        thread.start()

    @staticmethod
    def _kill_winword() -> None:
        import subprocess

        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "winword.exe", "/T"],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            log.exception("taskkill winword failed")

    def _read_html_sync(self) -> str:
        from PySide6.QtCore import QEventLoop

        result: dict[str, str] = {"html": ""}
        loop = QEventLoop()

        def _on_html(html: str) -> None:
            result["html"] = html or ""
            loop.quit()

        try:
            self._view.page().toHtml(_on_html)
        except Exception:
            log.exception("toHtml call failed")
            return ""

        guard = QTimer(self._view)
        guard.setSingleShot(True)
        guard.timeout.connect(loop.quit)
        guard.start(3000)

        loop.exec()
        guard.stop()

        if not result["html"]:
            log.warning("toHtml returned empty")
        return result["html"]

    def _doc_to_html(self, doc_path: str) -> Optional[str]:
        out_dir = tempfile.mkdtemp(prefix="word_app_")
        base = os.path.splitext(os.path.basename(doc_path))[0]
        html_path = os.path.join(out_dir, f"{base}.html")

        word = None
        doc = None
        try:
            log.info("_doc_to_html: DispatchEx Word.Application")
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0
            try:
                word.AutomationSecurity = _MSO_AUTOMATION_SECURITY_FORCE_DISABLE
            except Exception:
                pass

            doc = word.Documents.Open(
                FileName=doc_path,
                ConfirmConversions=False,
                ReadOnly=True,
                AddToRecentFiles=False,
            )
            doc.SaveAs2(FileName=html_path, FileFormat=_WD_FORMAT_FILTERED_HTML)
            log.info("_doc_to_html: saved %s", html_path)
            return html_path
        except Exception:
            log.exception("_doc_to_html: failed")
            return None
        finally:
            if doc is not None:
                try:
                    doc.Close(SaveChanges=False)
                except Exception:
                    pass
            if word is not None:
                try:
                    word.Quit()
                except Exception:
                    pass
            self._hide_leftover_word_windows()

    @staticmethod
    def _hide_leftover_word_windows() -> None:
        def _enum(hwnd: int, _: object) -> bool:
            try:
                if not win32gui.IsWindowVisible(hwnd):
                    return True
                if win32gui.GetClassName(hwnd) != "OpusApp":
                    return True
                win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(_enum, None)
        except Exception:
            pass
