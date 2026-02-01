# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt5.QtCore import Qt

from preview_record_tab import PreviewRecordTab
from calib_tab import CalibTab
from rectify_tab import RectifyTab   # 这里就是“Rectify+测距”的那个版本
from perception_3d_tab import Perception3DTab   # 三维感知模块

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stereo App (Preview/Record + Calibrate + Rectify + Measure + 3D Perception)")
        self.setGeometry(80, 80, 1600, 900)

        tabs = QTabWidget()
        tabs.addTab(PreviewRecordTab(), "预览+录制")
        tabs.addTab(CalibTab(), "标定")
        tabs.addTab(RectifyTab(), "Rectify+测距")   # 改个名字更直观
        tabs.addTab(Perception3DTab(), "三维感知")   # 新增三维感知标签页

        self.setCentralWidget(tabs)

if __name__ == "__main__":
    # 高清屏适配（建议这样写）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
