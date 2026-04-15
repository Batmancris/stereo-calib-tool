# -*- coding: utf-8 -*-
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "windows:fontengine=gdi")

from PyQt5.QtCore import Qt, qInstallMessageHandler
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QStyleFactory, QTabWidget

from preview_record_tab import PreviewRecordTab
from calib_tab import CalibTab
from rectify_tab import RectifyTab
from perception_3d_tab import Perception3DTab
from ui_theme import build_app_stylesheet


def _qt_message_filter(mode, context, message):
    text = str(message)
    if "CreateFontFaceFromHDC() failed" in text or "Fixedsys" in text:
        return
    sys.stderr.write(text + "\n")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stereo Vision Studio | 双目测量工作台")
        self.setGeometry(72, 60, 1680, 960)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(PreviewRecordTab(), "实时预览")
        tabs.addTab(CalibTab(), "双目标定")
        tabs.addTab(RectifyTab(), "双点测距")
        tabs.addTab(Perception3DTab(), "多点3D测量")
        self.setCentralWidget(tabs)


if __name__ == "__main__":
    qInstallMessageHandler(_qt_message_filter)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(build_app_stylesheet())
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
