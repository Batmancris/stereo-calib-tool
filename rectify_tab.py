# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from clickable_label import ClickableImageLabel
from config import RECORD_DIR, YAML_DIR
from ui_theme import create_page_header
from utils_common import bgr_to_qpixmap, read_mat
from utils_img import split_sbs


def reproject_point(u: float, v: float, d: float, reprojection_matrix: np.ndarray) -> np.ndarray:
    vec = reprojection_matrix @ np.array([float(u), float(v), float(d), 1.0], dtype=np.float64)
    w = vec[3]
    if abs(w) < 1e-12:
        raise RuntimeError("Invalid reprojection: W ~ 0")
    return np.array([vec[0] / w, vec[1] / w, vec[2] / w], dtype=np.float64)


class RectifyTab(QWidget):
    def __init__(self):
        super().__init__()

        self.reprojection_matrix: Optional[np.ndarray] = None
        self.image_width: Optional[int] = None
        self.image_height: Optional[int] = None
        self.map1x: Optional[np.ndarray] = None
        self.map1y: Optional[np.ndarray] = None
        self.map2x: Optional[np.ndarray] = None
        self.map2y: Optional[np.ndarray] = None

        self.video_capture: Optional[cv2.VideoCapture] = None
        self.is_playing: bool = False
        self.current_frame: Optional[np.ndarray] = None
        self.left_rectified: Optional[np.ndarray] = None
        self.right_rectified: Optional[np.ndarray] = None
        self.latest_left: Optional[np.ndarray] = None
        self.latest_right: Optional[np.ndarray] = None

        self.point_A_left: Optional[Tuple[float, float]] = None
        self.point_A_right: Optional[Tuple[float, float]] = None
        self.point_B_left: Optional[Tuple[float, float]] = None
        self.point_B_right: Optional[Tuple[float, float]] = None

        self._init_ui()

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._on_play_tick)
        self._apply_display_fps()

    def _init_ui(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(14)
        lay.addWidget(create_page_header(
            "\u53cc\u70b9 3D \u6d4b\u8ddd",
            "\u9002\u5408\u5feb\u901f\u9a8c\u8bc1\u6807\u5b9a\u8d28\u91cf\uff0c\u4f30\u8ba1\u4e24\u4e2a\u7279\u5f81\u70b9\u4e4b\u95f4\u7684\u4e09\u7ef4\u8ddd\u79bb\uff0c\u5e76\u5bf9\u5355\u5e27\u5b9e\u9a8c\u6570\u636e\u505a\u4eba\u5de5\u590d\u6838\u3002",
            accent="#5c9ecf",
        ))

        self._create_yaml_section(lay)
        self._create_video_section(lay)
        self._create_measure_section(lay)
        self._create_view_section(lay)
        self._create_log_section(lay)

        scroll_area.setWidget(container)
        main_lay = QVBoxLayout(self)
        main_lay.addWidget(scroll_area)

    def _create_yaml_section(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        self.yaml_edit = QLineEdit()
        self.yaml_edit.setPlaceholderText("\u9009\u62e9 stereo_calib_*.yaml")
        row.addWidget(self.yaml_edit)

        btn_pick = QPushButton("\u9009YAML")
        btn_pick.clicked.connect(self.pick_yaml)
        row.addWidget(btn_pick)

        btn_load = QPushButton("\u52a0\u8f7dYAML")
        btn_load.clicked.connect(self.load_yaml)
        row.addWidget(btn_load)
        layout.addLayout(row)

    def _create_video_section(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("\u9009\u62e9 record_*.avi")
        row.addWidget(self.video_edit)

        btn_pick = QPushButton("\u9009\u89c6\u9891")
        btn_pick.clicked.connect(self.pick_video)
        row.addWidget(btn_pick)

        btn_open = QPushButton("\u6253\u5f00\u89c6\u9891")
        btn_open.clicked.connect(self.open_video)
        row.addWidget(btn_open)

        self.btn_play = QPushButton("\u64ad\u653e")
        self.btn_play.clicked.connect(self.toggle_play)
        row.addWidget(self.btn_play)

        btn_step = QPushButton("\u6b65\u8fdb +1 \u5e27")
        btn_step.clicked.connect(self.step_one)
        row.addWidget(btn_step)

        btn_freeze = QPushButton("\u51bb\u7ed3\u6d4b\u91cf\u5e27")
        btn_freeze.clicked.connect(self.freeze_for_measure)
        row.addWidget(btn_freeze)
        layout.addLayout(row)

    def _create_measure_section(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel("\u5df2\u77e5\u957f\u5ea6(mm):"))
        self.len_mm = QDoubleSpinBox()
        self.len_mm.setRange(0.1, 5000.0)
        self.len_mm.setDecimals(2)
        self.len_mm.setValue(100.0)
        row.addWidget(self.len_mm)

        row.addWidget(QLabel("\u663e\u793aFPS:"))
        self.disp_fps = QSpinBox()
        self.disp_fps.setRange(1, 120)
        self.disp_fps.setValue(30)
        self.disp_fps.valueChanged.connect(self._apply_display_fps)
        row.addWidget(self.disp_fps)

        btn_clear = QPushButton("\u6e05\u9664\u70b9")
        btn_clear.clicked.connect(self.clear_points)
        row.addWidget(btn_clear)

        btn_calc = QPushButton("\u8ba1\u7b97 3D \u8ddd\u79bb")
        btn_calc.clicked.connect(self.compute_distance)
        row.addWidget(btn_calc)
        layout.addLayout(row)

    def _create_view_section(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()

        self.viewL = ClickableImageLabel("\u5de6\u56fe\uff1a\u6309 A\u5de6 -> B\u5de6 \u987a\u5e8f\u70b9\u51fb")
        self.viewL.setMinimumSize(640, 480)
        self.viewL.setObjectName("ImagePanel")
        self.viewL.clicked.connect(self.on_click_left)
        row.addWidget(self.viewL, 1)

        self.viewR = ClickableImageLabel("\u53f3\u56fe\uff1a\u6309 A\u53f3 -> B\u53f3 \u987a\u5e8f\u70b9\u51fb")
        self.viewR.setMinimumSize(640, 480)
        self.viewR.setObjectName("ImagePanel")
        self.viewR.clicked.connect(self.on_click_right)
        row.addWidget(self.viewR, 1)

        layout.addLayout(row)

    def _create_log_section(self, layout: QVBoxLayout) -> None:
        self.log = QTextEdit()
        self.log.setObjectName("LogPanel")
        self.log.setReadOnly(True)
        self.log.setFixedHeight(220)
        layout.addWidget(self.log)

    def log_add(self, message: str):
        self.log.append(message)

    def pick_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "\u9009\u62e9YAML", YAML_DIR, "YAML (*.yaml *.yml)")
        if path:
            self.yaml_edit.setText(path)

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "\u9009\u62e9\u89c6\u9891", RECORD_DIR, "Video (*.avi *.mp4 *.mkv)")
        if path:
            self.video_edit.setText(path)

    def load_yaml(self) -> None:
        yaml_path = self.yaml_edit.text().strip()
        if not yaml_path or not os.path.isfile(yaml_path):
            QMessageBox.warning(self, "\u63d0\u793a", "\u8bf7\u5148\u9009\u62e9 YAML \u6587\u4ef6")
            return

        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        self.image_width = int(fs.getNode("image_width").real())
        self.image_height = int(fs.getNode("image_height").real())
        k1 = read_mat(fs, "K1")
        d1 = read_mat(fs, "D1")
        k2 = read_mat(fs, "K2")
        d2 = read_mat(fs, "D2")
        r1 = read_mat(fs, "R1")
        p1 = read_mat(fs, "P1")
        r2 = read_mat(fs, "R2")
        p2 = read_mat(fs, "P2")
        self.reprojection_matrix = read_mat(fs, "Q")
        fs.release()

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(k1, d1, r1, p1, (self.image_width, self.image_height), cv2.CV_16SC2)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(k2, d2, r2, p2, (self.image_width, self.image_height), cv2.CV_16SC2)

        self.log_add(f"[OK] YAML\u52a0\u8f7d\u6210\u529f\uff1a\u56fe\u50cf\u5c3a\u5bf8 ({self.image_width}x{self.image_height})\uff0c\u6821\u6b63\u6620\u5c04\u5df2\u5c31\u7eea\u3002")
        QMessageBox.information(self, "\u63d0\u793a", "YAML \u52a0\u8f7d\u6210\u529f\uff0c\u53ef\u4ee5\u6253\u5f00\u89c6\u9891\u5f00\u59cb\u6d4b\u91cf\u3002")

    def open_video(self) -> None:
        video_path = self.video_edit.text().strip()
        if not video_path or not os.path.isfile(video_path):
            QMessageBox.warning(self, "\u63d0\u793a", "\u8bf7\u5148\u9009\u62e9\u89c6\u9891\u6587\u4ef6")
            return
        if self.map1x is None or self.reprojection_matrix is None:
            QMessageBox.warning(self, "\u63d0\u793a", "\u8bf7\u5148\u52a0\u8f7d YAML")
            return

        self._close_video_capture()
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            self.video_capture = None
            QMessageBox.warning(self, "\u9519\u8bef", "\u65e0\u6cd5\u6253\u5f00\u89c6\u9891")
            return

        self.is_playing = False
        self.btn_play.setText("\u64ad\u653e")
        self.clear_points()
        self.log_add("[OK] \u89c6\u9891\u5df2\u6253\u5f00\uff0c\u53ef\u4ee5\u64ad\u653e\u6216\u5355\u5e27\u51bb\u7ed3\u6d4b\u91cf\u3002")
        self._read_and_show_one_frame()

    def _apply_display_fps(self) -> None:
        fps = int(self.disp_fps.value())
        self.play_timer.setInterval(max(1, int(round(1000.0 / float(fps)))))
        if self.is_playing:
            self.play_timer.start()

    def toggle_play(self) -> None:
        if self.video_capture is None:
            QMessageBox.warning(self, "\u9519\u8bef", "\u65e0\u6cd5\u6253\u5f00\u89c6\u9891")
            return
        if self.map1x is None:
            QMessageBox.warning(self, "\u63d0\u793a", "\u8bf7\u5148\u52a0\u8f7d YAML")
            return

        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("\u6682\u505c")
            self.play_timer.start()
        else:
            self.btn_play.setText("\u64ad\u653e")
            self.play_timer.stop()

    def step_one(self) -> None:
        if self.video_capture is None:
            QMessageBox.warning(self, "\u9519\u8bef", "\u65e0\u6cd5\u6253\u5f00\u89c6\u9891")
            return
        if self.is_playing:
            self.is_playing = False
            self.btn_play.setText("\u64ad\u653e")
            self.play_timer.stop()
        self._read_and_show_one_frame()

    def freeze_for_measure(self) -> None:
        if self.left_rectified is None or self.right_rectified is None:
            QMessageBox.warning(self, "\u63d0\u793a", "\u8bf7\u5148\u6253\u5f00\u89c6\u9891\u5e76\u51bb\u7ed3\u4e00\u5e27\u518d\u6d4b\u91cf")
            return

        if self.is_playing:
            self.is_playing = False
            self.btn_play.setText("\u64ad\u653e")
            self.play_timer.stop()

        self.clear_points()
        self.log_add("[OK] \u5df2\u51bb\u7ed3\u5f53\u524d\u5e27\uff0c\u8bf7\u6309 A\u5de6 -> A\u53f3 -> B\u5de6 -> B\u53f3 \u987a\u5e8f\u70b9\u51fb\u3002")
        self._update_visualization()
        self._show_latest_frames()

    def _on_play_tick(self) -> None:
        if not self.is_playing:
            return
        if not self._read_and_show_one_frame():
            self.is_playing = False
            self.btn_play.setText("\u64ad\u653e")
            self.play_timer.stop()
            self.log_add("[INFO] \u89c6\u9891\u5df2\u64ad\u653e\u5230\u672b\u5c3e\u3002")

    def _read_and_show_one_frame(self) -> bool:
        if self.video_capture is None:
            return False
        ret, frame = self.video_capture.read()
        if not ret or frame is None:
            return False

        self.current_frame = frame
        self._rectify_current_frame()
        self._update_visualization()
        self._show_latest_frames()
        return True

    def _rectify_current_frame(self) -> None:
        left, right = split_sbs(self.current_frame, 0, 0)
        self.left_rectified = cv2.remap(left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        self.right_rectified = cv2.remap(right, self.map2x, self.map2y, cv2.INTER_LINEAR)

    def _close_video_capture(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
        self.is_playing = False
        if self.video_capture is not None:
            try:
                self.video_capture.release()
            except Exception:
                pass
        self.video_capture = None

    def clear_points(self) -> None:
        self.point_A_left = None
        self.point_A_right = None
        self.point_B_left = None
        self.point_B_right = None
        self.log_add("[INFO] \u5df2\u6e05\u9664\u70b9\u4f4d\uff0c\u8bf7\u6309 A\u5de6 -> A\u53f3 -> B\u5de6 -> B\u53f3 \u91cd\u65b0\u70b9\u51fb\u3002")

    def _update_visualization(self) -> None:
        if self.left_rectified is None or self.right_rectified is None:
            return

        left = self.left_rectified.copy()
        right = self.right_rectified.copy()

        step = max(1, self.image_height // 10)
        for y in range(step, self.image_height, step):
            cv2.line(left, (0, y), (left.shape[1] - 1, y), (0, 255, 0), 1)
            cv2.line(right, (0, y), (right.shape[1] - 1, y), (0, 255, 0), 1)

        def draw_pt(img: np.ndarray, pt: Optional[Tuple[float, float]], color: Tuple[int, int, int], label: str) -> None:
            if pt is None:
                return
            x, y = int(pt[0]), int(pt[1])
            cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(img, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        draw_pt(left, self.point_A_left, (0, 0, 255), "A_L")
        draw_pt(right, self.point_A_right, (0, 0, 255), "A_R")
        draw_pt(left, self.point_B_left, (255, 0, 0), "B_L")
        draw_pt(right, self.point_B_right, (255, 0, 0), "B_R")

        self.latest_left = left
        self.latest_right = right

    def _show_latest_frames(self) -> None:
        if self.latest_left is not None:
            self.viewL.set_image_pixmap(bgr_to_qpixmap(self.latest_left), self.latest_left.shape[1], self.latest_left.shape[0])
        if self.latest_right is not None:
            self.viewR.set_image_pixmap(bgr_to_qpixmap(self.latest_right), self.latest_right.shape[1], self.latest_right.shape[0])

    def on_click_left(self, u: float, v: float) -> None:
        if self.left_rectified is None:
            self.log_add("[WARN] \u5f53\u524d\u6ca1\u6709\u53ef\u70b9\u51fb\u7684\u6821\u6b63\u56fe\u50cf\u3002")
            return
        if self.is_playing:
            self.log_add("[HINT] \u8bf7\u5148\u51bb\u7ed3\u5f53\u524d\u5e27\uff0c\u518d\u8fdb\u884c\u70b9\u51fb\u6d4b\u91cf\u3002")
            return

        if self.point_A_left is None:
            self.point_A_left = (u, v)
            self.log_add(f"[\u70b9\u4f4d] A_L = {(u, v)}")
        elif self.point_B_left is None:
            self.point_B_left = (u, v)
            self.log_add(f"[\u70b9\u4f4d] B_L = {(u, v)}")
        else:
            self.log_add("[INFO] \u5de6\u56fe\u4e24\u4e2a\u70b9\u5df2\u8bb0\u5f55\u5b8c\u6210\u3002")
        self._update_visualization()
        self._show_latest_frames()

    def on_click_right(self, u: float, v: float) -> None:
        if self.right_rectified is None:
            self.log_add("[WARN] \u5f53\u524d\u6ca1\u6709\u53ef\u70b9\u51fb\u7684\u6821\u6b63\u56fe\u50cf\u3002")
            return
        if self.is_playing:
            self.log_add("[HINT] \u8bf7\u5148\u51bb\u7ed3\u5f53\u524d\u5e27\uff0c\u518d\u8fdb\u884c\u70b9\u51fb\u6d4b\u91cf\u3002")
            return

        if self.point_A_left is None:
            self.log_add("[HINT] \u8bf7\u5148\u5728\u5de6\u56fe\u70b9\u51fb A_L\u3002")
            return
        if self.point_A_right is None:
            self.point_A_right = (u, v)
            self.log_add(f"[\u70b9\u4f4d] A_R = {(u, v)}")
        elif self.point_B_left is None:
            self.log_add("[HINT] \u8bf7\u5148\u5728\u5de6\u56fe\u70b9\u51fb B_L\u3002")
            return
        elif self.point_B_right is None:
            self.point_B_right = (u, v)
            self.log_add(f"[\u70b9\u4f4d] B_R = {(u, v)}")
        else:
            self.log_add("[INFO] \u53f3\u56fe\u4e24\u4e2a\u70b9\u5df2\u8bb0\u5f55\u5b8c\u6210\u3002")
        self._update_visualization()
        self._show_latest_frames()

    def compute_distance(self) -> None:
        if self.reprojection_matrix is None or self.image_width is None:
            QMessageBox.warning(self, "\u63d0\u793a", "\u8bf7\u5148\u52a0\u8f7d YAML\uff0c\u786e\u4fdd Q \u77e9\u9635\u53ef\u7528\u3002")
            return
        if any(x is None for x in [self.point_A_left, self.point_A_right, self.point_B_left, self.point_B_right]):
            QMessageBox.warning(self, "\u63d0\u793a", "\u8bf7\u6309 A\u5de6 -> A\u53f3 -> B\u5de6 -> B\u53f3 \u987a\u5e8f\u70b9\u6ee1\u56db\u4e2a\u5bf9\u5e94\u70b9\u3002")
            return

        uL_A, vL_A = map(float, self.point_A_left)
        uR_A, vR_A = map(float, self.point_A_right)
        uL_B, vL_B = map(float, self.point_B_left)
        uR_B, vR_B = map(float, self.point_B_right)

        def v_check(name: str, v_left: float, v_right: float) -> None:
            dv = abs(v_left - v_right)
            if dv > 2.0:
                self.log_add(f"[WARN] {name}: |vL-vR|={dv:.2f}px\uff0c\u5de6\u53f3\u70b9\u7684\u884c\u504f\u5dee\u8f83\u5927\u3002")

        v_check("A", vL_A, vR_A)
        v_check("B", vL_B, vR_B)

        vA = 0.5 * (vL_A + vR_A)
        vB = 0.5 * (vL_B + vR_B)
        dA = uL_A - uR_A
        dB = uL_B - uR_B
        if dA <= 0 or dB <= 0:
            self.log_add(f"[ERR] disparity<=0: dA={dA:.3f}, dB={dB:.3f}\uff0c\u5bf9\u5e94\u5173\u7cfb\u53ef\u80fd\u9009\u53cd\u6216\u70b9\u4f4d\u4e0d\u51c6\u3002")
            return

        try:
            pA = reproject_point(uL_A, vA, dA, self.reprojection_matrix)
            pB = reproject_point(uL_B, vB, dB, self.reprojection_matrix)
        except Exception as exc:
            self.log_add(f"[\u9519\u8bef] \u4e09\u7ef4\u91cd\u6295\u5f71\u5931\u8d25: {exc}")
            return

        dist = float(np.linalg.norm(pA - pB))
        gt = float(self.len_mm.value())
        err = dist - gt
        rel = (err / gt) * 100.0 if gt > 1e-9 else 0.0

        self.log_add(f"[3D] A\u70b9(mm) = {pA.round(2)}   \u89c6\u5dee dA={dA:.2f}px")
        self.log_add(f"[3D] B\u70b9(mm) = {pB.round(2)}   \u89c6\u5dee dB={dB:.2f}px")
        self.log_add(f"[\u6d4b\u91cf] \u8ddd\u79bb = {dist:.2f} mm | \u8bbe\u5b9a\u503c = {gt:.2f} mm | \u8bef\u5dee = {err:+.2f} mm ({rel:+.2f}%)")

        QMessageBox.information(
            self,
            "\u6d4b\u91cf\u7ed3\u679c",
            f"\u6d4b\u5f97\u8ddd\u79bb: {dist:.2f} mm\n\u8bbe\u5b9a\u957f\u5ea6: {gt:.2f} mm\n\u8bef\u5dee: {err:+.2f} mm ({rel:+.2f}%)",
        )
