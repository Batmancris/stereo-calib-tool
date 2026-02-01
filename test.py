# -*- coding: utf-8 -*-
"""
实时双目(拼接)视频显示 + 固定分割 + 显示侧亮度/对比度/伽马调节
目标：3840x1080 @ 30fps 且尽量使用 MJPG（压缩）避免卡顿

关键改动：
1) 采集端：优先尝试 MSMF 获取 MJPG；失败再退回 DSHOW
2) 显示端：先 resize 到 label 尺寸再转 QImage（大幅减轻 CPU）
3) 打印真实 FPS（每秒一次），避免 cap.get(FPS) 虚标
"""

import sys
import time
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, cam_id: int):
        super().__init__()
        self.cam_id = cam_id
        self.running = True

        # 固定分割参数（默认0）
        self.SPLIT_OFFSET = 0
        self.SPLIT_GAP = 0

        # 目标采集参数（规格书：3840x1080@30 MJPEG）
        self.TARGET_W = 3840
        self.TARGET_H = 1080
        self.TARGET_FPS = 30
        self.TARGET_FOURCC = "MJPG"  # 期望拿到 MJPEG

        # 真实 FPS 统计
        self._cnt = 0
        self._t0 = time.time()

    def set_split_offset(self, v: int):
        self.SPLIT_OFFSET = int(v)

    def set_split_gap(self, v: int):
        self.SPLIT_GAP = max(0, int(v))

    @staticmethod
    def _fourcc_to_str(fourcc_int: int) -> str:
        # 有些后端返回的 fourcc 可能带符号位，先 mask
        x = fourcc_int & 0xFFFFFFFF
        return "".join([chr((x >> 8 * i) & 0xFF) for i in range(4)])

    def _try_open_with_backend(self, backend: int, backend_name: str):
        cap = cv2.VideoCapture(self.cam_id, backend)
        if not cap.isOpened():
            return None

        # 强制期望格式与分辨率
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.TARGET_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.TARGET_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.TARGET_H)
        cap.set(cv2.CAP_PROP_FPS, self.TARGET_FPS)

        # 读回确认
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = self._fourcc_to_str(fourcc)
        print(f"[CAP-{backend_name}] {w}x{h} @ {fps:.1f}fps FOURCC={fourcc_str}")

        return cap

    def _open_camera_best(self):
        # 优先 MSMF（更容易拿到 MJPG），不行再用 DSHOW
        for backend, name in [(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW")]:
            cap = self._try_open_with_backend(backend, name)
            if cap is None:
                continue

            # 若成功拿到 MJPG 且分辨率正确，直接用
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = self._fourcc_to_str(fourcc)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if (w, h) == (self.TARGET_W, self.TARGET_H) and fourcc_str.strip("\x00") in ("MJPG", "MJPG"):
                print("[OK] Using MJPG compressed stream.")
                return cap

            # 如果分辨率对但 fourcc 不是 MJPG，也先保留（但可能会卡）
            if (w, h) == (self.TARGET_W, self.TARGET_H):
                print(f"[WARN] Got target resolution but FOURCC={fourcc_str} (may be heavy).")
                return cap

            # 否则释放继续尝试下一个后端
            cap.release()

        return None

    def run(self):
        cap = self._open_camera_best()
        if cap is None or not cap.isOpened():
            print("[ERR] Camera open/config failed.")
            return

        # 尽量降低缓存延迟（并非所有后端支持）
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            if w / float(h) > 2.0:
                cut = w // 2 + self.SPLIT_OFFSET
                cut = max(1, min(w - 1, cut))

                gap = self.SPLIT_GAP
                left_end = max(1, cut - gap // 2)
                right_start = min(w - 1, cut + (gap - gap // 2))

                left = frame[:, :left_end]
                right = frame[:, right_start:]
            else:
                left = frame
                right = frame

            self.frame_signal.emit(left, right)

            # 真实 FPS 统计（每秒打印一次）
            self._cnt += 1
            t1 = time.time()
            if t1 - self._t0 >= 1.0:
                real_fps = self._cnt / (t1 - self._t0)
                print(f"[REAL FPS] {real_fps:.1f}")
                self._cnt = 0
                self._t0 = t1

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时视频显示")
        self.setGeometry(100, 100, 1400, 550)

        self.thread = None
        self.current_left = None
        self.current_right = None

        self._gamma_table = None

        self.init_ui()
        self.find_cameras()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("摄像头:"))
        self.cam_combo = QComboBox()
        ctrl.addWidget(self.cam_combo)

        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self.toggle)
        ctrl.addWidget(self.start_btn)

        self.snap_btn = QPushButton("截图")
        self.snap_btn.clicked.connect(self.snapshot)
        ctrl.addWidget(self.snap_btn)

        ctrl.addWidget(QLabel("对比度α:"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.2, 3.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(1.0)
        ctrl.addWidget(self.alpha_spin)

        ctrl.addWidget(QLabel("亮度β:"))
        self.beta_spin = QSpinBox()
        self.beta_spin.setRange(-255, 255)
        self.beta_spin.setValue(0)
        ctrl.addWidget(self.beta_spin)

        ctrl.addWidget(QLabel("伽马γ:"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.2, 3.0)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.valueChanged.connect(self._update_gamma_table)
        ctrl.addWidget(self.gamma_spin)

        ctrl.addWidget(QLabel("分割偏移(px):"))
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(-800, 800)
        self.offset_spin.setValue(0)
        self.offset_spin.valueChanged.connect(self.on_split_param_changed)
        ctrl.addWidget(self.offset_spin)

        ctrl.addWidget(QLabel("分割间隙(px):"))
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(0, 400)
        self.gap_spin.setValue(0)
        self.gap_spin.valueChanged.connect(self.on_split_param_changed)
        ctrl.addWidget(self.gap_spin)

        layout.addLayout(ctrl)

        display = QHBoxLayout()
        self.left_label = QLabel("左")
        self.left_label.setMinimumSize(640, 480)
        self.left_label.setStyleSheet("border: 1px solid gray; background: black;")
        display.addWidget(self.left_label)

        self.right_label = QLabel("右")
        self.right_label.setMinimumSize(640, 480)
        self.right_label.setStyleSheet("border: 1px solid gray; background: black;")
        display.addWidget(self.right_label)

        layout.addLayout(display)

        self._update_gamma_table()

    def _update_gamma_table(self):
        gamma = float(self.gamma_spin.value())
        if abs(gamma - 1.0) < 1e-6:
            self._gamma_table = None
            return
        inv = 1.0 / gamma
        table = (np.linspace(0, 255, 256) / 255.0) ** inv
        self._gamma_table = np.clip(table * 255.0, 0, 255).astype(np.uint8)

    def adjust_display(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return img
        alpha = float(self.alpha_spin.value())
        beta = int(self.beta_spin.value())
        out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        if self._gamma_table is not None:
            out = cv2.LUT(out, self._gamma_table)
        return out

    def _to_pixmap_for_label(self, bgr_img: np.ndarray, label: QLabel) -> QPixmap:
        """关键优化：先 resize 到 label 尺寸附近，再转 QImage（否则 4K 每帧太重）"""
        if bgr_img is None:
            return QPixmap()

        # label 的目标尺寸（按 KeepAspectRatio 逻辑缩放）
        target_w = max(1, label.width())
        target_h = max(1, label.height())

        h, w = bgr_img.shape[:2]
        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # 先缩小再转色彩
        small = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        h1, w1, c1 = rgb.shape
        qimg = QImage(rgb.data, w1, h1, c1 * w1, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def find_cameras(self):
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()

        # 用 DSHOW 探测即可
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.cam_combo.addItem(f"Camera {i}")
                cap.release()

        self.cam_combo.blockSignals(False)

    def on_split_param_changed(self):
        if self.thread and self.thread.isRunning():
            self.thread.set_split_offset(self.offset_spin.value())
            self.thread.set_split_gap(self.gap_spin.value())

    def toggle(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread = None
            self.start_btn.setText("开始")
        else:
            if self.cam_combo.count() > 0:
                cam_id = int(self.cam_combo.currentText().split()[-1])
                self.start(cam_id)

    def start(self, cam_id: int):
        if self.thread and self.thread.isRunning():
            self.thread.stop()

        self.thread = VideoThread(cam_id)
        self.thread.frame_signal.connect(self.update_frames)

        self.thread.set_split_offset(self.offset_spin.value())
        self.thread.set_split_gap(self.gap_spin.value())

        self.thread.start()
        self.start_btn.setText("停止")

    def update_frames(self, left: np.ndarray, right: np.ndarray):
        # 保存原始（用于截图/算法）
        self.current_left = left
        self.current_right = right

        # 显示侧调节
        left_disp = self.adjust_display(left)
        right_disp = self.adjust_display(right)

        # 关键优化：先 resize 再转 QImage
        self.left_label.setPixmap(self._to_pixmap_for_label(left_disp, self.left_label))
        self.right_label.setPixmap(self._to_pixmap_for_label(right_disp, self.right_label))

    def snapshot(self):
        if self.current_left is None or self.current_right is None:
            return

        cv2.imwrite("left.jpg", self.current_left)
        cv2.imwrite("right.jpg", self.current_right)

        # 若要保存显示增强后版本，取消注释：
        # cv2.imwrite("left_display.jpg", self.adjust_display(self.current_left))
        # cv2.imwrite("right_display.jpg", self.adjust_display(self.current_right))

        print("截图已保存: left.jpg, right.jpg")

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
