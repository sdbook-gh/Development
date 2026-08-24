"""Embed Microsoft Word in-place via QAxWidget OLE + pywin32 COM."""

from __future__ import annotations

import os
from typing import Optional

import win32com.client
import win32con
import win32gui
from PySide6.QtAxContainer import QAxWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QVBoxLayout, QWidget

from app_logging import get_logger

log = get_logger("word_host.ole")

# Office MsoAutomationSecurity
_MSO_AUTOMATION_SECURITY_FORCE_DISABLE = 3

# WinEvent hooks removed: OUTOFCONTEXT callbacks touching COM/Qt caused UI freezes.


class WordOleHost:
    """Host a Word document inside a Qt widget using ActiveX/OLE."""

    def __init__(self, container: QWidget) -> None:
        self.container = container
        self.container.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._ax = QAxWidget(container)
        self._ax.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self._ax)

        self._path: Optional[str] = None
        self._doc: Optional[object] = None
        self._word_hwnd: Optional[int] = None
        self._hide_timers: list[QTimer] = []
        self._suppressing = False
        log.debug("WordOleHost created")

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def is_loaded(self) -> bool:
        return bool(self._ax.control())

    def create_blank(self) -> None:
        """Embed a blank Word.Document as an in-form editing surface (paste-ready)."""
        log.info("create_blank: start")
        self.close()

        log.info("create_blank: setControl(Word.Document)")
        if not self._ax.setControl("Word.Document"):
            log.error("create_blank: setControl failed")
            raise RuntimeError(
                "无法创建 Word 编辑空间：Word.Document ActiveX 初始化失败。"
                "请确认已安装 Microsoft Word。"
            )
        log.info("create_blank: setControl ok, control=%r", self._ax.control())

        self._path = None
        self._harden_ax_application()
        # Avoid GetActiveObject during blank init — it often deadlocks with OLE.
        self._suppress_standalone_ui(enum_windows=False)
        self._schedule_suppress_passes()
        log.info("create_blank: done")

    def load(self, path: str) -> None:
        abs_path = os.path.abspath(path)
        log.info("load: start path=%s", abs_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(abs_path)

        self.close()

        log.info("load: setControl(file)")
        if not self._ax.setControl(abs_path):
            log.warning("load: setControl(file) failed, fallback Word.Document Open")
            self._load_via_word_document(abs_path)
        else:
            log.info("load: setControl(file) ok")

        self._path = abs_path
        self._harden_ax_application()
        self._suppress_standalone_ui(enum_windows=False)
        self._schedule_pywin32_attach(abs_path)
        self._schedule_suppress_passes()
        log.info("load: scheduled attach/suppress")

    def save_as(self, path: str) -> str:
        log.info("save_as: request path=%s", path)
        if not self.is_loaded:
            raise RuntimeError("没有可保存的 Word 文档，请先创建编辑区或打开文件。")

        abs_path = os.path.abspath(path)
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        ext = os.path.splitext(abs_path)[1].lower()
        if ext == ".doc":
            file_format = 0
        elif ext == ".rtf":
            file_format = 6
        else:
            if ext != ".docx":
                abs_path = abs_path + ".docx"
            file_format = 16

        self._ensure_pywin32_doc()
        if self._doc is not None:
            log.info("save_as: via pywin32 SaveAs2 format=%s", file_format)
            self._doc.SaveAs2(FileName=abs_path, FileFormat=file_format)
        else:
            log.info("save_as: via QAxWidget SaveAs2/SaveAs")
            saved = self._ax.dynamicCall(
                "SaveAs2(const QString&, int)",
                abs_path,
                file_format,
            )
            if saved is False:
                self._ax.dynamicCall("SaveAs(const QString&)", abs_path)

        self._path = abs_path
        self._suppress_standalone_ui(enum_windows=True)
        log.info("save_as: ok -> %s", abs_path)
        return abs_path

    def close(self) -> None:
        log.info("close: start path=%s has_doc=%s", self._path, self._doc is not None)
        self._cancel_hide_timers()

        # Closing COM ActiveDocument while QAx still hosts it often freezes.
        # Only release the ActiveX control; skip doc.Close for blank / embedded docs.
        if self._doc is not None and self._path:
            try:
                log.debug("close: Save before release skipped; releasing refs only")
            except Exception:
                pass
        self._doc = None
        self._word_hwnd = None
        self._path = None
        try:
            self._ax.clear()
            log.info("close: ax.clear done")
        except Exception:
            log.exception("close: ax.clear failed")

    def _load_via_word_document(self, abs_path: str) -> None:
        if not self._ax.setControl("Word.Document"):
            raise RuntimeError(
                "OLE 嵌入失败：无法创建 Word.Document ActiveX。"
                "请确认已安装 Microsoft Word。"
            )

        self._harden_ax_application()
        app = self._ax.querySubObject("Application")
        if app is None:
            raise RuntimeError("OLE 嵌入失败：无法获取 Word.Application。")

        docs = app.querySubObject("Documents")
        if docs is None:
            raise RuntimeError("OLE 嵌入失败：无法获取 Documents 集合。")

        opened = docs.querySubObject(
            "Open(const QString&, bool, bool, bool)",
            abs_path,
            False,
            True,
            False,
        )
        if opened is None:
            opened = docs.querySubObject("Open(const QString&)", abs_path)
        if opened is None:
            raise RuntimeError(f"OLE 嵌入失败：无法打开文档\n{abs_path}")
        log.info("_load_via_word_document: opened")

    def _schedule_pywin32_attach(self, abs_path: Optional[str]) -> None:
        timer = QTimer(self.container)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._delayed_attach(abs_path))
        timer.start(200)
        self._hide_timers.append(timer)
        log.debug("scheduled pywin32 attach in 200ms")

    def _delayed_attach(self, abs_path: Optional[str]) -> None:
        log.debug("delayed_attach: abs_path=%s current=%s", abs_path, self._path)
        if abs_path is not None and self._path != abs_path:
            return
        if abs_path is None:
            # Blank document: do not GetActiveObject here (freeze risk).
            log.debug("delayed_attach: skip for blank document")
            return
        self._attach_pywin32(abs_path)
        self._suppress_standalone_ui(enum_windows=True)

    def _attach_pywin32(self, abs_path: Optional[str]) -> None:
        self._doc = None
        if not abs_path:
            log.debug("_attach_pywin32: no path, skip")
            return
        try:
            log.info("_attach_pywin32: GetObject(%s)", abs_path)
            self._doc = win32com.client.GetObject(abs_path)
            app = self._doc.Application
            self._apply_automation_hardening(app)
            self._word_hwnd = int(app.Hwnd)
            log.info("_attach_pywin32: ok hwnd=%s", self._word_hwnd)
        except Exception:
            log.exception("_attach_pywin32: failed")
            self._doc = None
            self._word_hwnd = None

    def _ensure_pywin32_doc(self) -> None:
        if self._doc is not None:
            return
        if self._path:
            self._attach_pywin32(self._path)
            return
        # Last resort for blank unsaved doc — may hang on some Office builds.
        log.warning("_ensure_pywin32_doc: trying GetActiveObject for blank doc")
        try:
            word = win32com.client.GetActiveObject("Word.Application")
            self._apply_automation_hardening(word)
            self._doc = word.ActiveDocument
            self._word_hwnd = int(word.Hwnd)
            log.info("_ensure_pywin32_doc: GetActiveObject ok")
        except Exception:
            log.exception("_ensure_pywin32_doc: GetActiveObject failed")
            self._doc = None

    def _harden_ax_application(self) -> None:
        app = self._ax.querySubObject("Application")
        if app is None:
            log.debug("_harden_ax_application: no Application yet")
            return
        try:
            app.setProperty("AutomationSecurity", _MSO_AUTOMATION_SECURITY_FORCE_DISABLE)
        except Exception:
            log.debug("AutomationSecurity set failed", exc_info=True)
        try:
            app.setProperty("DisplayAlerts", 0)
            app.setProperty("ScreenUpdating", False)
            app.setProperty("Visible", False)
        except Exception:
            log.debug("Visible/DisplayAlerts set failed", exc_info=True)
        try:
            hwnd = int(app.property("Hwnd"))
            if hwnd:
                self._word_hwnd = hwnd
                log.debug("Application.Hwnd=%s", hwnd)
        except Exception:
            pass

    @staticmethod
    def _apply_automation_hardening(app: object) -> None:
        for name, value in (
            ("AutomationSecurity", _MSO_AUTOMATION_SECURITY_FORCE_DISABLE),
            ("DisplayAlerts", 0),
            ("ScreenUpdating", False),
            ("Visible", False),
        ):
            try:
                setattr(app, name, value)
            except Exception:
                pass

    def _suppress_standalone_ui(self, *, enum_windows: bool = True) -> None:
        if self._suppressing:
            log.debug("suppress: re-entrant skip")
            return
        self._suppressing = True
        try:
            log.debug("suppress: start enum_windows=%s", enum_windows)
            self._harden_ax_application()
            self._hide_via_pywin32()
            if self._word_hwnd:
                self._hide_hwnd_if_toplevel(self._word_hwnd)
            if enum_windows:
                self._hide_orphan_word_hwnds()
                self._dismiss_vba_environment_dialogs()
            log.debug("suppress: done")
        finally:
            self._suppressing = False

    def _hide_via_pywin32(self) -> None:
        if self._doc is None:
            return
        try:
            app = self._doc.Application
            self._apply_automation_hardening(app)
            self._word_hwnd = int(app.Hwnd)
            self._hide_hwnd_if_toplevel(self._word_hwnd)
        except Exception:
            log.debug("_hide_via_pywin32 failed", exc_info=True)

    def _hide_orphan_word_hwnds(self) -> None:
        container_hwnd = int(self.container.winId())
        ax_hwnd = int(self._ax.winId())

        def _enum(hwnd: int, _: object) -> bool:
            try:
                class_name = win32gui.GetClassName(hwnd)
            except Exception:
                return True
            if class_name == "OpusApp":
                if not (
                    self._is_descendant(hwnd, container_hwnd)
                    or self._is_descendant(hwnd, ax_hwnd)
                ):
                    self._hide_toplevel_word_frame(hwnd)
            elif class_name == "#32770" and self._looks_like_vba_dialog(hwnd):
                self._close_dialog(hwnd)
            return True

        try:
            win32gui.EnumWindows(_enum, None)
        except Exception:
            log.debug("EnumWindows failed", exc_info=True)

    def _hide_hwnd_if_toplevel(self, hwnd: int) -> None:
        if not hwnd or not win32gui.IsWindow(hwnd):
            return
        container_hwnd = int(self.container.winId())
        ax_hwnd = int(self._ax.winId())
        if self._is_descendant(hwnd, container_hwnd) or self._is_descendant(hwnd, ax_hwnd):
            return
        self._hide_toplevel_word_frame(hwnd)

    def _hide_toplevel_word_frame(self, hwnd: int) -> None:
        log.debug("hide toplevel hwnd=%s", hwnd)
        try:
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            ex_style = (ex_style | win32con.WS_EX_TOOLWINDOW) & ~win32con.WS_EX_APPWINDOW
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)
        except Exception:
            pass
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
        except Exception:
            pass

    @staticmethod
    def _looks_like_vba_dialog(hwnd: int) -> bool:
        try:
            title = win32gui.GetWindowText(hwnd) or ""
        except Exception:
            return False
        lower = title.lower()
        if "visual basic" in lower or "vba" in lower:
            return True
        if "无法创建" in title and ("visual" in lower or "basic" in lower or "vba" in lower):
            return True
        if "无法创建" in title and "环境" in title:
            return True
        return False

    @staticmethod
    def _close_dialog(hwnd: int) -> None:
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        except Exception:
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
            except Exception:
                pass

    @staticmethod
    def _is_descendant(hwnd: int, ancestor: int) -> bool:
        if not hwnd or not ancestor:
            return False
        current = hwnd
        while current:
            if current == ancestor:
                return True
            current = win32gui.GetParent(current)
        return False

    def _dismiss_vba_environment_dialogs(self) -> None:
        def _enum(hwnd: int, _: object) -> bool:
            try:
                if win32gui.GetClassName(hwnd) == "#32770" and self._looks_like_vba_dialog(hwnd):
                    log.info("dismiss VBA dialog hwnd=%s title=%r", hwnd, win32gui.GetWindowText(hwnd))
                    self._close_dialog(hwnd)
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(_enum, None)
        except Exception:
            pass

    def _schedule_suppress_passes(self) -> None:
        # Fewer passes; first without EnumWindows heavy work if already done lightly.
        for delay_ms in (150, 500, 1200):
            timer = QTimer(self.container)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self._suppress_standalone_ui(enum_windows=True))
            timer.start(delay_ms)
            self._hide_timers.append(timer)
        log.debug("scheduled suppress passes")

    def _cancel_hide_timers(self) -> None:
        for timer in self._hide_timers:
            timer.stop()
            timer.deleteLater()
        self._hide_timers.clear()
