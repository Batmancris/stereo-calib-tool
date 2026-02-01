# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QTextEdit, QMessageBox, QDoubleSpinBox, QSpinBox
)
from PyQt5.QtGui import QImage, QPixmap

from utils_img import split_sbs
from config import RECORD_DIR, YAML_DIR
from clickable_label import ClickableImageLabel


def read_mat(fs, name):
    node = fs.getNode(name)
    if node.empty():
        raise RuntimeError(f"YAML missing: {name}")
    return node.mat()


def bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def reproject_point(u, v, d, Q):
    vec = Q @ np.array([float(u), float(v), float(d), 1.0], dtype=np.float64)
    W = vec[3]
    if abs(W) < 1e-12:
        raise RuntimeError("Invalid reprojection: W ~ 0")
    X = vec[0] / W
    Y = vec[1] / W
    Z = vec[2] / W
    return np.array([X, Y, Z], dtype=np.float64)


class RectifyTab(QWidget):
    def __init__(self):
        super().__init__()

        self.Q = None
        self.w = None
        self.h = None
        self.map1x = None
        self.map1y = None
        self.map2x = None
        self.map2y = None

        self.cap = None
        self.playing = False
        self.cur_frame = None
        self.lrect = None
        self.rrect = None

        # 左右各自的“显示图”(带辅助线和点)
        self.latest_L = None
        self.latest_R = None

        # 点：全部用“各自图像坐标”(u,v)，不再用拼接坐标
        self.A_L = None
        self.A_R = None
        self.B_L = None
        self.B_R = None

        self._init_ui()

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._on_play_tick)
        self._apply_display_fps()

    def _init_ui(self):
        lay = QVBoxLayout(self)

        # YAML
        r1 = QHBoxLayout()
        self.yaml_edit = QLineEdit()
        self.yaml_edit.setPlaceholderText("选择 stereo_calib_*.yaml")
        r1.addWidget(self.yaml_edit)
        b1 = QPushButton("选YAML")
        b1.clicked.connect(self.pick_yaml)
        r1.addWidget(b1)
        bload = QPushButton("加载YAML(预计算矫正)")
        bload.clicked.connect(self.load_yaml)
        r1.addWidget(bload)
        lay.addLayout(r1)

        # Video
        r2 = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("选择一个 record_*.avi")
        r2.addWidget(self.video_edit)

        b2 = QPushButton("选视频")
        b2.clicked.connect(self.pick_video)
        r2.addWidget(b2)

        b_open = QPushButton("打开视频")
        b_open.clicked.connect(self.open_video)
        r2.addWidget(b_open)

        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self.toggle_play)
        r2.addWidget(self.btn_play)

        b_step = QPushButton("步进 +1帧")
        b_step.clicked.connect(self.step_one)
        r2.addWidget(b_step)

        b_freeze = QPushButton("冻结为测距帧")
        b_freeze.clicked.connect(self.freeze_for_measure)
        r2.addWidget(b_freeze)

        lay.addLayout(r2)

        # Measure controls
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("已知长度(mm):"))
        self.len_mm = QDoubleSpinBox()
        self.len_mm.setRange(0.1, 5000.0)
        self.len_mm.setDecimals(2)
        self.len_mm.setValue(100.0)
        r3.addWidget(self.len_mm)

        r3.addWidget(QLabel("显示FPS:"))
        self.disp_fps = QSpinBox()
        self.disp_fps.setRange(1, 120)
        self.disp_fps.setValue(30)
        self.disp_fps.valueChanged.connect(self._apply_display_fps)
        r3.addWidget(self.disp_fps)

        btn_clear = QPushButton("清除点")
        btn_clear.clicked.connect(self.clear_points)
        r3.addWidget(btn_clear)

        btn_calc = QPushButton("计算 3D 距离/误差")
        btn_calc.clicked.connect(self.compute_distance)
        r3.addWidget(btn_calc)

        lay.addLayout(r3)

        # View：左右分离显示（各自缩放/拖拽）
        view_row = QHBoxLayout()

        self.viewL = ClickableImageLabel("左图（滚轮缩放/拖拽）\n按顺序点：A左→B左")
        self.viewL.setMinimumSize(640, 480)
        self.viewL.setStyleSheet("border:1px solid gray; background:black;")
        self.viewL.clicked.connect(self.on_click_left)
        view_row.addWidget(self.viewL, 1)

        self.viewR = ClickableImageLabel("右图（滚轮缩放/拖拽）\n按顺序点：A右→B右")
        self.viewR.setMinimumSize(640, 480)
        self.viewR.setStyleSheet("border:1px solid gray; background:black;")
        self.viewR.clicked.connect(self.on_click_right)
        view_row.addWidget(self.viewR, 1)

        lay.addLayout(view_row)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(220)
        lay.addWidget(self.log)

    def log_add(self, s):
        self.log.append(s)

    def pick_yaml(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择YAML", YAML_DIR, "YAML (*.yaml *.yml)")
        if p:
            self.yaml_edit.setText(p)

    def pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择视频", RECORD_DIR, "Video (*.avi *.mp4 *.mkv)")
        if p:
            self.video_edit.setText(p)

    def load_yaml(self):
        yp = self.yaml_edit.text().strip()
        if not yp or not os.path.isfile(yp):
            QMessageBox.warning(self, "错误", "请选择有效 YAML")
            return

        fs = cv2.FileStorage(yp, cv2.FILE_STORAGE_READ)
        self.w = int(fs.getNode("image_width").real())
        self.h = int(fs.getNode("image_height").real())
        K1 = read_mat(fs, "K1"); D1 = read_mat(fs, "D1")
        K2 = read_mat(fs, "K2"); D2 = read_mat(fs, "D2")
        R1 = read_mat(fs, "R1"); P1 = read_mat(fs, "P1")
        R2 = read_mat(fs, "R2"); P2 = read_mat(fs, "P2")
        self.Q = read_mat(fs, "Q")
        fs.release()

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (self.w, self.h), cv2.CV_16SC2)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (self.w, self.h), cv2.CV_16SC2)

        self.log_add(f"[OK] YAML loaded. image=({self.w}x{self.h}), rect maps ready.")
        QMessageBox.information(self, "完成", "YAML 加载完成，矫正映射已预计算。")

    def open_video(self):
        vp = self.video_edit.text().strip()
        if not vp or not os.path.isfile(vp):
            QMessageBox.warning(self, "错误", "请选择有效视频")
            return
        if self.map1x is None or self.Q is None:
            QMessageBox.warning(self, "错误", "请先加载 YAML（预计算矫正映射）")
            return

        self._close_cap()
        self.cap = cv2.VideoCapture(vp)
        if not self.cap.isOpened():
            self.cap = None
            QMessageBox.warning(self, "错误", "无法打开视频")
            return

        self.playing = False
        self.btn_play.setText("▶ 播放")
        self.clear_points()

        self.log_add("[OK] video opened. 播放/步进到清晰帧后暂停，再点“冻结为测距帧”。")
        self._read_and_show_one_frame()

    def _apply_display_fps(self):
        fps = int(self.disp_fps.value())
        interval_ms = max(1, int(round(1000.0 / float(fps))))
        self.play_timer.setInterval(interval_ms)
        if self.playing:
            self.play_timer.start()

    def toggle_play(self):
        if self.cap is None:
            QMessageBox.warning(self, "错误", "请先打开视频")
            return
        if self.map1x is None:
            QMessageBox.warning(self, "错误", "请先加载 YAML")
            return

        self.playing = not self.playing
        if self.playing:
            self.btn_play.setText("⏸ 暂停")
            self.play_timer.start()
        else:
            self.btn_play.setText("▶ 播放")
            self.play_timer.stop()

    def step_one(self):
        if self.cap is None:
            QMessageBox.warning(self, "错误", "请先打开视频")
            return
        if self.playing:
            self.playing = False
            self.btn_play.setText("▶ 播放")
            self.play_timer.stop()
        self._read_and_show_one_frame()

    def freeze_for_measure(self):
        if self.lrect is None or self.rrect is None:
            QMessageBox.warning(self, "错误", "当前没有可用帧（请先播放/步进）")
            return

        if self.playing:
            self.playing = False
            self.btn_play.setText("▶ 播放")
            self.play_timer.stop()

        self.clear_points()
        self.log_add("[OK] 已冻结当前帧为测距帧。请按 A左→A右→B左→B右 点击取点。")
        self._update_vis_lr()
        self._show_latest_lr()

    def _on_play_tick(self):
        if not self.playing:
            return
        ok = self._read_and_show_one_frame()
        if not ok:
            self.playing = False
            self.btn_play.setText("▶ 播放")
            self.play_timer.stop()
            self.log_add("[INFO] 视频到末尾，已停止。")

    def _read_and_show_one_frame(self) -> bool:
        if self.cap is None:
            return False
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False

        self.cur_frame = frame
        self._rectify_current_frame()
        self._update_vis_lr()
        self._show_latest_lr()
        return True

    def _rectify_current_frame(self):
        left, right = split_sbs(self.cur_frame, 0, 0)
        self.lrect = cv2.remap(left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        self.rrect = cv2.remap(right, self.map2x, self.map2y, cv2.INTER_LINEAR)

    def _close_cap(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
        self.playing = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def clear_points(self):
        self.A_L = self.A_R = self.B_L = self.B_R = None
        self.log_add("[INFO] cleared points. 点击顺序：A左→A右→B左→B右")

    def _update_vis_lr(self):
        if self.lrect is None or self.rrect is None:
            return

        L = self.lrect.copy()
        R = self.rrect.copy()

        # 画水平线（用于检验矫正后同名点应同一水平线）
        step = max(1, self.h // 10)
        for y in range(step, self.h, step):
            cv2.line(L, (0, y), (L.shape[1] - 1, y), (0, 255, 0), 1)
            cv2.line(R, (0, y), (R.shape[1] - 1, y), (0, 255, 0), 1)

        def draw_pt(img, pt, color, label):
            if pt is None:
                return
            x, y = int(pt[0]), int(pt[1])
            cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(img, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        draw_pt(L, self.A_L, (0, 0, 255), "A_L")
        draw_pt(R, self.A_R, (0, 0, 255), "A_R")
        draw_pt(L, self.B_L, (255, 0, 0), "B_L")
        draw_pt(R, self.B_R, (255, 0, 0), "B_R")

        self.latest_L = L
        self.latest_R = R

    def _show_latest_lr(self):
        if self.latest_L is not None:
            pmL = bgr_to_qpixmap(self.latest_L)
            self.viewL.set_image_pixmap(pmL, self.latest_L.shape[1], self.latest_L.shape[0])
        if self.latest_R is not None:
            pmR = bgr_to_qpixmap(self.latest_R)
            self.viewR.set_image_pixmap(pmR, self.latest_R.shape[1], self.latest_R.shape[0])

    # ---------- 点击逻辑：左右分别处理 ----------
    def on_click_left(self, u, v):
        if self.lrect is None:
            self.log_add("[WARN] 还没有帧可点")
            return
        if self.playing:
            self.log_add("[HINT] 建议先暂停并点“冻结为测距帧”，再取点。")
            return

        if self.A_L is None:
            self.A_L = (u, v)
            self.log_add(f"[PT] A_L = {(u, v)}")
        elif self.B_L is None:
            self.B_L = (u, v)
            self.log_add(f"[PT] B_L = {(u, v)}")
        else:
            self.log_add("[INFO] 左图点已选完（A_L、B_L）。")
        self._update_vis_lr()
        self._show_latest_lr()

    def on_click_right(self, u, v):
        if self.rrect is None:
            self.log_add("[WARN] 还没有帧可点")
            return
        if self.playing:
            self.log_add("[HINT] 建议先暂停并点“冻结为测距帧”，再取点。")
            return

        if self.A_L is None:
            self.log_add("[HINT] 第1步请先在左图点 A_L")
            return

        if self.A_R is None:
            self.A_R = (u, v)
            self.log_add(f"[PT] A_R = {(u, v)}")
        elif self.B_L is None:
            self.log_add("[HINT] 第3步请先在左图点 B_L")
            return
        elif self.B_R is None:
            self.B_R = (u, v)
            self.log_add(f"[PT] B_R = {(u, v)}")
        else:
            self.log_add("[INFO] 右图点已选完（A_R、B_R）。")
        self._update_vis_lr()
        self._show_latest_lr()

    def compute_distance(self):
        if self.Q is None or self.w is None:
            QMessageBox.warning(self, "错误", "请先加载 YAML 并打开视频/冻结帧")
            return
        if any(x is None for x in [self.A_L, self.A_R, self.B_L, self.B_R]):
            QMessageBox.warning(self, "错误", "请先按顺序点击四个点：A左→A右→B左→B右")
            return

        uL_A, vL_A = float(self.A_L[0]), float(self.A_L[1])
        uR_A, vR_A = float(self.A_R[0]), float(self.A_R[1])
        uL_B, vL_B = float(self.B_L[0]), float(self.B_L[1])
        uR_B, vR_B = float(self.B_R[0]), float(self.B_R[1])

        def v_check(name, vL, vR):
            dv = abs(vL - vR)
            if dv > 2.0:
                self.log_add(f"[WARN] {name}: |vL-vR|={dv:.2f}px 偏大：要么点错对应点，要么矫正不够好。")

        v_check("A", vL_A, vR_A)
        v_check("B", vL_B, vR_B)

        vA = 0.5 * (vL_A + vR_A)
        vB = 0.5 * (vL_B + vR_B)

        dA = uL_A - uR_A
        dB = uL_B - uR_B
        if dA <= 0 or dB <= 0:
            self.log_add(f"[ERR] disparity<=0: dA={dA:.3f}, dB={dB:.3f}（通常是对应点点错/左右反了）")
            return

        try:
            PA = reproject_point(uL_A, vA, dA, self.Q)
            PB = reproject_point(uL_B, vB, dB, self.Q)
        except Exception as e:
            self.log_add(f"[ERR] reprojection failed: {e}")
            return

        dist = float(np.linalg.norm(PA - PB))
        gt = float(self.len_mm.value())
        err = dist - gt
        rel = (err / gt) * 100.0 if gt > 1e-9 else 0.0

        self.log_add(f"[3D] A(mm) = {PA.round(2)}   dA={dA:.2f}px")
        self.log_add(f"[3D] B(mm) = {PB.round(2)}   dB={dB:.2f}px")
        self.log_add(f"[MEAS] distance = {dist:.2f} mm | GT = {gt:.2f} mm | error = {err:+.2f} mm ({rel:+.2f}%)")

        QMessageBox.information(
            self, "测距结果",
            f"测得距离: {dist:.2f} mm\n真实长度: {gt:.2f} mm\n误差: {err:+.2f} mm ({rel:+.2f}%)"
        )
