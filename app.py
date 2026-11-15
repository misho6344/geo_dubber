"""GeoDubber entry point."""
import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler

from config.settings import APP_NAME, sub_dir


def setup_logging() -> None:
    h = RotatingFileHandler(sub_dir("logs") / "geodubber.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[h, logging.StreamHandler()])


def main() -> int:
    setup_logging()
    log = logging.getLogger("app")
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    from gui.styles import STYLESHEET
    app.setStyleSheet(STYLESHEET)

    def hook(et, ev, tb):
        log.error("Unhandled exception\n%s", "".join(traceback.format_exception(et, ev, tb)))
        QMessageBox.critical(None, APP_NAME, f"Unexpected error:\n{ev}\n\nDetails were saved to logs/geodubber.log")
    sys.excepthook = hook

    from core.ffmpeg_utils import ffmpeg_exe
    from core.errors import GeoDubberError
    try:
        ffmpeg_exe()
    except GeoDubberError as e:
        QMessageBox.warning(None, APP_NAME, e.user_message)
    from gui.main_window import MainWindow
    w = MainWindow()
    w.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
