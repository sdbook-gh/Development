"""PySide6 + pywin32 Word OLE editor entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app_logging import setup_logging, get_logger
from ui import MainWindow


def main() -> int:
    log_path = setup_logging()
    log = get_logger("main")
    log.info("=== Word OLE Editor starting ===")
    log.info("Log file: %s", log_path)

    app = QApplication(sys.argv)
    app.setApplicationName("Word OLE Editor")
    window = MainWindow()
    window.show()
    log.info("Main window shown, entering event loop")
    code = app.exec()
    log.info("=== exiting code=%s ===", code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
