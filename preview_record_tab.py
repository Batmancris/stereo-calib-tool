# -*- coding: utf-8 -*-
import os
import time
import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt5.QtCore import QTimer

from config import RECORD_DIR, SPLIT_OFFSET, SPLIT_GAP
from ffmpeg_io import FFmpegPreviewThread
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
        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        container = QWidget()
        lay = QVBoxLayout(container)

        ctrl = QHBoxLayout()
        self.btn_preview = QPushButton("开始预览")
        self.btn_preview.clicked.connect(self.toggle_preview)
        ctrl.addWidget(self.btn_preview)

        self.btn_record = QPushButton("开始录制")
        self.btn_record.clicked.connect(self.toggle_record)
        ctrl.addWidget(self.btn_record)

        self.btn_shot = QPushButton("截图(预览分辨率)")
        self.btn_shot.clicked.connect(self.snapshot)
        ctrl.addWidget(self.btn_shot)

        lay.addLayout(ctrl)

        disp = QHBoxLayout()
        self.lab_l = QLabel("Left")
        self.lab_l.setMinimumSize(640, 480)
        self.lab_l.setStyleSheet("border:1px solid gray; background:black;")
        disp.addWidget(self.lab_l)

        self.lab_r = QLabel("Right")
        self.lab_r.setMinimumSize(640, 480)
        self.lab_r.setStyleSheet("border:1px solid gray; background:black;")
        disp.addWidget(self.lab_r)
        lay.addLayout(disp)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(140)
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
            self.btn_preview.setText("停止预览")
        else:
            self.thread.stop_stream()
            self.btn_preview.setText("开始预览")
            self.btn_record.setText("开始录制")

    def toggle_record(self):
        if self.thread.proc is None:
            path = self._make_record_path()
            self.latest = None
            self.thread.start_stream(record=True, record_path=path)
            self.btn_preview.setText("停止预览")
            self.btn_record.setText("停止录制")
            self.on_log(f"[REC] {path}")
            return

        # 已经在跑
        if not self.thread.recording:
            self.thread.stop_stream()
            time.sleep(0.2)
            path = self._make_record_path()
            self.latest = None
            self.thread.start_stream(record=True, record_path=path)
            self.btn_record.setText("停止录制")
            self.btn_preview.setText("停止预览")
            self.on_log(f"[REC] {path}")
        else:
            self.thread.stop_stream()
            time.sleep(0.2)
            self.latest = None
            self.thread.start_stream(record=False)
            self.btn_record.setText("开始录制")
            self.btn_preview.setText("停止预览")
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
