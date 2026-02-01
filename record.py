# -*- coding: utf-8 -*-
"""
Qt 内嵌预览 + 录制（MJPEG 3840x1080@30）
修复：
- 录制时预览“四宫格/颜色乱” => pipe 输出 rawvideo 的实际 pix_fmt 变化（常见 bgr0/bgra=4Bpp）
- 解决：对 pipe:1 这一路输出强制 -vcodec rawvideo -pix_fmt bgr24 -s WxH
录制仍然 map 0:v 直接 copy，不重编码

录制路径：E:\research\0\stero
文件名：record_时间戳.avi（防覆盖）
"""

import sys
import os
import time
import subprocess
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap


# ===================== 参数区 =====================
DEVICE_NAME = "DECXIN Camera"
SBS_W, SBS_H = 3840, 1080
FPS = 30

# 预览输出分辨率（越大越清晰，越小越省）
PREVIEW_W, PREVIEW_H = 1920, 540

# 分割微调
SPLIT_OFFSET = 0
SPLIT_GAP = 0

FFMPEG_EXE = r"C:\Users\12548\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
RECORD_DIR = r"E:\research\0\stero"
# =================================================

if not os.path.isfile(FFMPEG_EXE):
    raise FileNotFoundError(f"FFMPEG_EXE 不存在：{FFMPEG_EXE}")
print(f"[FFMPEG] {FFMPEG_EXE}")
os.makedirs(RECORD_DIR, exist_ok=True)


def split_sbs(frame_bgr, offset=0, gap=0):
    """把 SBS 拼接图分割成左右眼"""
    h, w = frame_bgr.shape[:2]
    cut = w // 2 + int(offset)
    cut = max(1, min(w - 1, cut))
    gap = max(0, int(gap))
    left_end = max(1, cut - gap // 2)
    right_start = min(w - 1, cut + (gap - gap // 2))
    left = frame_bgr[:, :left_end]
    right = frame_bgr[:, right_start:]
    return left, right


class FFmpegReader(QThread):
    frame_signal = pyqtSignal(np.ndarray)  # 预览用：缩放后的整张 SBS BGR

    def __init__(self):
        super().__init__()
        self.proc = None
        self.running = False
        self.recording = False
        self.record_path = None

    def start_stream(self, record=False, record_path=None):
        if self.proc is not None:
            return

        self.recording = bool(record)
        self.record_path = record_path

        if self.recording and self.record_path:
            os.makedirs(os.path.dirname(self.record_path) or ".", exist_ok=True)

        # 输入：dshow + mjpeg + 3840x1080@30
        base_in = [
            FFMPEG_EXE,
            "-hide_banner",
            "-loglevel", "warning",
            "-f", "dshow",
            "-video_size", f"{SBS_W}x{SBS_H}",
            "-framerate", str(FPS),
            "-vcodec", "mjpeg",
            "-i", f"video={DEVICE_NAME}",
        ]

        if not self.recording:
            # 只预览：scale -> pipe(rawvideo bgr24)
            cmd = base_in + [
                "-vf", f"scale={PREVIEW_W}:{PREVIEW_H},format=bgr24",

                # ---- 强制 pipe 输出参数（防止 pix_fmt/stride 变化导致四宫格）----
                "-map", "0:v",
                "-an",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{PREVIEW_W}x{PREVIEW_H}",
                "-f", "rawvideo",
                "pipe:1",
            ]
        else:
            # 预览 + 录制：
            # 预览走滤镜输出到 pipe，录制直接 copy 原始 0:v 到文件
            cmd = base_in + [
                "-filter_complex", f"[0:v]scale={PREVIEW_W}:{PREVIEW_H},format=bgr24[vprev]",

                # ---- 预览输出（pipe）----
                "-map", "[vprev]",
                "-an",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{PREVIEW_W}x{PREVIEW_H}",
                "-f", "rawvideo",
                "pipe:1",

                # ---- 录制输出（原始流 copy）----
                "-map", "0:v",
                "-c:v", "copy",
                "-f", "avi",
                self.record_path,
            ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        self.running = True
        self.start()

    def stop_stream(self):
        self.running = False
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = None

    def run(self):
        # pipe 输出帧固定为 PREVIEW_W x PREVIEW_H x 3(bgr24)
        bytes_per_frame = PREVIEW_W * PREVIEW_H * 3
        buf = b""

        last_print = time.time()
        cnt = 0

        while self.running and self.proc and self.proc.stdout:
            chunk = self.proc.stdout.read(bytes_per_frame - len(buf))
            if not chunk:
                if self.proc and self.proc.stderr:
                    try:
                        err = self.proc.stderr.read().decode(errors="ignore")
                        if err.strip():
                            print("[FFMPEG-ERR]\n" + err)
                    except Exception:
                        pass
                break

            buf += chunk
            if len(buf) < bytes_per_frame:
                continue

            frame_bytes = buf[:bytes_per_frame]
            buf = buf[bytes_per_frame:]

            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((PREVIEW_H, PREVIEW_W, 3))
            self.frame_signal.emit(frame)

            cnt += 1
            t = time.time()
            if t - last_print >= 1.0:
                print(f"[PREVIEW FPS] {cnt/(t-last_print):.1f}")
                cnt = 0
                last_print = t

        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt 预览 + 录制 (ffmpeg)")
        self.setGeometry(100, 100, 1400, 600)

        self.reader = FFmpegReader()
        self.reader.frame_signal.connect(self.on_new_frame)

        self.latest_frame = None
        self.current_left = None
        self.current_right = None

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_view)
        self.timer.start(30)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

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

        layout.addLayout(ctrl)

        disp = QHBoxLayout()
        self.left_label = QLabel("Left")
        self.left_label.setMinimumSize(640, 480)
        self.left_label.setStyleSheet("border:1px solid gray; background:black;")
        disp.addWidget(self.left_label)

        self.right_label = QLabel("Right")
        self.right_label.setMinimumSize(640, 480)
        self.right_label.setStyleSheet("border:1px solid gray; background:black;")
        disp.addWidget(self.right_label)

        layout.addLayout(disp)

    def _make_record_path(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(RECORD_DIR, f"record_{ts}.avi")

    def toggle_preview(self):
        if self.reader.proc is None:
            self.latest_frame = None
            self.reader.start_stream(record=False)
            self.btn_preview.setText("停止预览")
        else:
            self.reader.stop_stream()
            self.btn_preview.setText("开始预览")
            self.btn_record.setText("开始录制")

    def toggle_record(self):
        if self.reader.proc is None:
            path = self._make_record_path()
            self.latest_frame = None
            self.reader.start_stream(record=True, record_path=path)
            self.btn_preview.setText("停止预览")
            self.btn_record.setText("停止录制")
            print(f"[REC] {path}")
            return

        if not self.reader.recording:
            self.reader.stop_stream()
            time.sleep(0.2)
            path = self._make_record_path()
            self.latest_frame = None
            self.reader.start_stream(record=True, record_path=path)
            self.btn_record.setText("停止录制")
            self.btn_preview.setText("停止预览")
            print(f"[REC] {path}")
        else:
            self.reader.stop_stream()
            time.sleep(0.2)
            self.latest_frame = None
            self.reader.start_stream(record=False)
            self.btn_record.setText("开始录制")
            self.btn_preview.setText("停止预览")
            print("[REC] stop")

    def on_new_frame(self, frame_bgr):
        self.latest_frame = frame_bgr

    def refresh_view(self):
        if self.latest_frame is None:
            return
        frame = self.latest_frame
        left, right = split_sbs(frame, SPLIT_OFFSET, SPLIT_GAP)
        self.current_left = left
        self.current_right = right
        self.left_label.setPixmap(self.to_pixmap(left, self.left_label))
        self.right_label.setPixmap(self.to_pixmap(right, self.right_label))

    def to_pixmap(self, bgr, label):
        target_w, target_h = max(1, label.width()), max(1, label.height())
        h, w = bgr.shape[:2]
        s = min(target_w / w, target_h / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, nw, nh, 3 * nw, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def snapshot(self):
        if self.current_left is None or self.current_right is None:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        lp = os.path.join(RECORD_DIR, f"shot_left_{ts}.jpg")
        rp = os.path.join(RECORD_DIR, f"shot_right_{ts}.jpg")
        cv2.imwrite(lp, self.current_left)
        cv2.imwrite(rp, self.current_right)
        print(f"[SHOT] {lp} / {rp}")

    def closeEvent(self, event):
        self.reader.stop_stream()
        event.accept()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
