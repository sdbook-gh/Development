"""Main application window: HTML/Chromium editor + Word save."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from word_host import WordHtmlHost
from app_logging import get_logger

log = get_logger("ui.main_window")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        log.info("MainWindow init")
        self.setWindowTitle("Word HTML Editor")
        self.resize(1100, 800)

        self._path_label = QLabel("正在初始化编辑区…")
        self._path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._path_label.setStyleSheet("color: #444; padding: 4px 8px;")

        self._hint_label = QLabel(
            "在下方编辑区可直接粘贴（Ctrl+V）从 Word 复制的内容；保存为 .docx / .rtf（纯 Python 转换，无需 Word）。"
            " 日志: logs/word_app.log"
        )
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet("color: #666; padding: 0 8px 4px 8px;")

        self._host_container = QWidget()
        self._host_container.setObjectName("wordHostContainer")
        self._host_container.setStyleSheet(
            "#wordHostContainer { background: #ffffff; border: 1px solid #b0b0b0; }"
        )
        self._host_container.setMinimumSize(400, 300)

        self._save_button = QPushButton("保存到 Word 文件…")
        self._save_button.setMinimumHeight(36)
        self._save_button.clicked.connect(self.save_document)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("当前:"))
        path_row.addWidget(self._path_label, stretch=1)
        layout.addLayout(path_row)
        layout.addWidget(self._hint_label)
        layout.addWidget(self._host_container, stretch=1)

        save_row = QHBoxLayout()
        save_row.addStretch(1)
        save_row.addWidget(self._save_button)
        layout.addLayout(save_row)

        self.setCentralWidget(central)

        self._host = WordHtmlHost(self._host_container)
        self._build_toolbar()
        QTimer.singleShot(0, self.new_blank_document)
        log.info("MainWindow init done; blank scheduled")

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        new_action = QAction("新建空白", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_blank_document)
        toolbar.addAction(new_action)

        open_action = QAction("打开 Word…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_word_file)
        toolbar.addAction(open_action)

        save_action = QAction("保存…", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_document)
        toolbar.addAction(save_action)

        close_action = QAction("关闭并新建空白", self)
        close_action.triggered.connect(self.close_document)
        toolbar.addAction(close_action)

    def new_blank_document(self) -> None:
        log.info("UI: new_blank_document")
        self._path_label.setText("正在创建编辑区…")
        try:
            self._host.create_blank()
        except Exception as exc:
            log.exception("UI: create_blank failed")
            QMessageBox.critical(self, "初始化失败", f"无法创建编辑空间:\n{exc}")
            self._path_label.setText("编辑区不可用")
            self._save_button.setEnabled(False)
            return
        self._save_button.setEnabled(True)
        self._path_label.setText("空白编辑区（可粘贴，未保存）")
        log.info("UI: blank ready")

    def open_word_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Word 文件",
            "",
            "Word 文档 (*.docx *.doc *.docm *.rtf);;所有文件 (*.*)",
        )
        if not path:
            return
        log.info("UI: open_word_file %s", path)
        self._path_label.setText("正在打开（Word→HTML）…")
        self._save_button.setEnabled(False)

        def _done(_html_path: str) -> None:
            self._save_button.setEnabled(True)
            self._path_label.setText(path)
            log.info("UI: open done")

        def _fail(msg: str) -> None:
            log.exception("UI: load failed: %s", msg)
            QMessageBox.critical(self, "打开失败", f"无法加载 Word 文件:\n{msg}")
            self._path_label.setText("打开失败")
            self._save_button.setEnabled(True)

        try:
            self._host.load(path, _done, _fail)
        except Exception as exc:
            log.exception("UI: load dispatch failed")
            QMessageBox.critical(self, "打开失败", f"无法加载 Word 文件:\n{exc}")
            self._path_label.setText("打开失败")
            self._save_button.setEnabled(True)

    def save_document(self) -> None:
        if not self._host.is_loaded:
            QMessageBox.information(self, "提示", "当前没有可保存的文档。")
            return

        suggested = self._host.path or "未命名.docx"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 Word 文件",
            suggested,
            "Word 文档 (*.docx);;RTF (*.rtf)",
        )
        if not path:
            return
        log.info("UI: save_document %s", path)
        self._path_label.setText("正在保存（HTML→Word）…")
        self._save_button.setEnabled(False)
        QApplication.processEvents()
        try:
            saved = self._host.save_as(path)
        except Exception as exc:
            log.exception("UI: save failed")
            QMessageBox.critical(self, "保存失败", f"无法保存 Word 文件:\n{exc}")
            self._path_label.setText("保存失败")
            self._save_button.setEnabled(True)
            return
        self._save_button.setEnabled(True)
        self._path_label.setText(saved)
        QMessageBox.information(self, "保存成功", f"已保存到:\n{saved}")
        log.info("UI: save ok %s", saved)

    def close_document(self) -> None:
        log.info("UI: close_document -> new blank")
        self.new_blank_document()

    def closeEvent(self, event: QCloseEvent) -> None:
        log.info("UI: closeEvent")
        self._host.close()
        super().closeEvent(event)
