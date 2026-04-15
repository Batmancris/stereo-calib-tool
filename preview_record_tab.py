# -*- coding: utf-8 -*-
import os
import time

import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QTextEdit, QVBoxLayout, QWidget

from config import RECORD_DIR, SPLIT_GAP, SPLIT_OFFSET
from ffmpeg_io import FFmpegPreviewThread
from ui_theme import create_page_header
from utils_img import split_sbs, to_pixmap_fit


class PreviewRecordTab(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = FFmpegPreviewThread()
        self.thread.frame_signal.connect(self.on_frame)
        self.thread.log_signal.connect(self.on_log)

        self.latest = None
        self.left = None
        self.right = None

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(30)

    def init_ui(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(14)
        lay.addWidget(create_page_header(
            "\u5b9e\u65f6\u9884\u89c8\u4e0e\u91c7\u96c6",
            "\u7528\u4e8e\u68c0\u67e5\u53cc\u76ee\u753b\u9762\u3001\u5feb\u901f\u622a\u56fe\u548c\u5f55\u5236\u5b9e\u9a8c\u89c6\u9891\u3002\u5de6\u4fa7\u4fdd\u6301\u5b9e\u65f6\u9884\u89c8\uff0c\u53f3\u4fa7\u4fdd\u6301\u6210\u5bf9\u89c6\u56fe\uff0c\u65b9\u4fbf\u540e\u7eed\u6807\u5b9a\u4e0e\u6d4b\u91cf\u3002",
            accent="#6fa98d",
        ))

        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        self.btn_preview = QPushButton("\u5f00\u59cb\u9884\u89c8")
        self.btn_preview.clicked.connect(self.toggle_preview)
        ctrl.addWidget(self.btn_preview)

        self.btn_record = QPushButton("\u5f00\u59cb\u5f55\u5236")
        self.btn_record.clicked.connect(self.toggle_record)
        ctrl.addWidget(self.btn_record)

        self.btn_shot = QPushButton("\u622a\u56fe\u4fdd\u5b58")
        self.btn_shot.clicked.connect(self.snapshot)
        ctrl.addWidget(self.btn_shot)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        disp = QHBoxLayout()
        disp.setSpacing(14)

        self.lab_l = QLabel("????")
        self.lab_l.setObjectName("ImagePanel")
        self.lab_l.setMinimumSize(640, 480)
        disp.addWidget(self.lab_l)

        self.lab_r = QLabel("????")
        self.lab_r.setObjectName("ImagePanel")
        self.lab_r.setMinimumSize(640, 480)
        disp.addWidget(self.lab_r)

        lay.addLayout(disp)

        self.log = QTextEdit()
        self.log.setObjectName("LogPanel")
        self.log.setReadOnly(True)
        self.log.setFixedHeight(150)
        lay.addWidget(self.log)

        scroll_area.setWidget(container)
        main_lay = QVBoxLayout(self)
        main_lay.addWidget(scroll_area)

    def _make_record_path(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(RECORD_DIR, f"record_{ts}.avi")

    def on_log(self, s: str):
        self.log.append(s)

    def on_frame(self, frame_bgr):
        self.latest = frame_bgr

    def refresh(self):
        if self.latest is None:
            return
        self.left, self.right = split_sbs(self.latest, SPLIT_OFFSET, SPLIT_GAP)
        self.lab_l.setPixmap(to_pixmap_fit(self.left, self.lab_l))
        self.lab_r.setPixmap(to_pixmap_fit(self.right, self.lab_r))

    def toggle_preview(self):
        if self.thread.proc is None:
            self.latest = None
            self.thread.start_stream(record=False)
            self.btn_preview.setText("\u505c\u6b62\u9884\u89c8")
        else:
            self.thread.stop_stream()
            self.btn_preview.setText("\u505c\u6b62\u9884\u89c8")
            self.btn_record.setText("\u5f00\u59cb\u5f55\u5236")

    def toggle_record(self):
        if self.thread.proc is None:
            path = self._make_record_path()
            self.latest = None
            self.thread.start_stream(record=True, record_path=path)
            self.btn_preview.setText("\u505c\u6b62\u9884\u89c8")
            self.btn_record.setText("\u505c\u6b62\u5f55\u5236")
            self.on_log(f"[REC] {path}")
            return

        if not self.thread.recording:
            self.thread.stop_stream()
            time.sleep(0.2)
            path = self._make_record_path()
            self.latest = None
            self.thread.start_stream(record=True, record_path=path)
            self.btn_record.setText("\u505c\u6b62\u5f55\u5236")
            self.btn_preview.setText("\u505c\u6b62\u9884\u89c8")
            self.on_log(f"[REC] {path}")
        else:
            self.thread.stop_stream()
            time.sleep(0.2)
            self.latest = None
            self.thread.start_stream(record=False)
            self.btn_record.setText("\u5f00\u59cb\u5f55\u5236")
            self.btn_preview.setText("\u505c\u6b62\u9884\u89c8")
            self.on_log("[REC] stop")

    def snapshot(self):
        if self.left is None or self.right is None:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        lp = os.path.join(RECORD_DIR, f"shot_left_{ts}.jpg")
        rp = os.path.join(RECORD_DIR, f"shot_right_{ts}.jpg")
        cv2.imwrite(lp, self.left)
        cv2.imwrite(rp, self.right)
        self.on_log(f"[SHOT] {lp} / {rp}")
