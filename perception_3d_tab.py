# -*- coding: utf-8 -*-
import csv
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QFileDialog,
    QGroupBox,
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
from config import RECORD_DIR, SPLIT_GAP, SPLIT_OFFSET, YAML_DIR
from utils_common import bgr_to_qpixmap, read_mat
from utils_img import split_sbs


@dataclass
class PointObservation:
    point_id: int
    left_xy: Optional[Tuple[float, float]] = None
    right_xy: Optional[Tuple[float, float]] = None
    xyz: Optional[np.ndarray] = None
    status: str = "pending"
    confidence: float = 0.0


class Perception3DTab(QWidget):
    def __init__(self):
        super().__init__()
        self.image_width: Optional[int] = None
        self.image_height: Optional[int] = None
        self.map1x: Optional[np.ndarray] = None
        self.map1y: Optional[np.ndarray] = None
        self.map2x: Optional[np.ndarray] = None
        self.map2y: Optional[np.ndarray] = None
        self.proj_left: Optional[np.ndarray] = None
        self.proj_right: Optional[np.ndarray] = None

        self.video_capture: Optional[cv2.VideoCapture] = None
        self.video_fps: float = 30.0
        self.frame_index: int = -1
        self.is_playing: bool = False

        self.current_frame: Optional[np.ndarray] = None
        self.left_rectified: Optional[np.ndarray] = None
        self.right_rectified: Optional[np.ndarray] = None
        self.latest_left: Optional[np.ndarray] = None
        self.latest_right: Optional[np.ndarray] = None

        self.point_count: int = 15
        self.current_points: List[PointObservation] = []
        self.saved_frames: Dict[int, List[PointObservation]] = {}
        self.previous_saved_points: Optional[List[PointObservation]] = None
        self.previous_left_gray: Optional[np.ndarray] = None
        self.selected_point_index: int = 0
        self.initialized: bool = False

        self._init_ui()
        self._reset_current_points()

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._on_play_tick)
        self._apply_display_fps()

    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        lay = QVBoxLayout(container)
        self._create_yaml_section(lay)
        self._create_video_section(lay)
        self._create_tracking_section(lay)
        self._create_view_section(lay)
        self._create_log_section(lay)
        scroll.setWidget(container)
        root = QVBoxLayout(self)
        root.addWidget(scroll)

    def _create_yaml_section(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        self.yaml_edit = QLineEdit()
        self.yaml_edit.setPlaceholderText("选择 stereo_calib_*.yaml")
        row.addWidget(self.yaml_edit)
        btn_pick = QPushButton("选YAML")
        btn_pick.clicked.connect(self.pick_yaml)
        row.addWidget(btn_pick)
        btn_load = QPushButton("加载YAML")
        btn_load.clicked.connect(self.load_yaml)
        row.addWidget(btn_load)
        layout.addLayout(row)

    def _create_video_section(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("选择一个 record_*.avi")
        row.addWidget(self.video_edit)
        btn_pick = QPushButton("选视频")
        btn_pick.clicked.connect(self.pick_video)
        row.addWidget(btn_pick)
        btn_open = QPushButton("打开视频")
        btn_open.clicked.connect(self.open_video)
        row.addWidget(btn_open)
        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self.toggle_play)
        row.addWidget(self.btn_play)
        btn_step = QPushButton("步进 +1 帧")
        btn_step.clicked.connect(self.step_one)
        row.addWidget(btn_step)
        btn_freeze = QPushButton("冻结当前帧")
        btn_freeze.clicked.connect(self.freeze_current_frame)
        row.addWidget(btn_freeze)
        row.addWidget(QLabel("显示FPS:"))
        self.disp_fps = QSpinBox()
        self.disp_fps.setRange(1, 120)
        self.disp_fps.setValue(20)
        self.disp_fps.valueChanged.connect(self._apply_display_fps)
        row.addWidget(self.disp_fps)
        layout.addLayout(row)

    def _create_tracking_section(self, layout: QVBoxLayout):
        group = QGroupBox("高精度关键点追踪")
        outer = QVBoxLayout(group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("关键点数:"))
        self.point_count_spin = QSpinBox()
        self.point_count_spin.setRange(2, 50)
        self.point_count_spin.setValue(self.point_count)
        self.point_count_spin.valueChanged.connect(self.on_point_count_changed)
        row1.addWidget(self.point_count_spin)
        row1.addWidget(QLabel("当前点:"))
        self.point_index_spin = QSpinBox()
        self.point_index_spin.setRange(1, self.point_count)
        self.point_index_spin.setValue(1)
        self.point_index_spin.valueChanged.connect(self.on_selected_point_changed)
        row1.addWidget(self.point_index_spin)
        btn_prev = QPushButton("上一点")
        btn_prev.clicked.connect(lambda: self._shift_selected_point(-1))
        row1.addWidget(btn_prev)
        btn_next = QPushButton("下一点")
        btn_next.clicked.connect(lambda: self._shift_selected_point(1))
        row1.addWidget(btn_next)
        btn_init = QPushButton("开始首帧初始化")
        btn_init.clicked.connect(self.start_manual_init)
        row1.addWidget(btn_init)
        btn_clear = QPushButton("清当前点")
        btn_clear.clicked.connect(self.clear_selected_point)
        row1.addWidget(btn_clear)
        btn_clear_frame = QPushButton("清当前帧点")
        btn_clear_frame.clicked.connect(self.clear_current_points)
        row1.addWidget(btn_clear_frame)
        outer.addLayout(row1)

        row2 = QHBoxLayout()
        btn_save = QPushButton("保存当前帧")
        btn_save.clicked.connect(self.save_current_frame)
        row2.addWidget(btn_save)
        btn_track = QPushButton("自动跟踪下一帧")
        btn_track.clicked.connect(self.track_next_frame)
        row2.addWidget(btn_track)
        btn_csv = QPushButton("导出CSV")
        btn_csv.clicked.connect(self.export_csv)
        row2.addWidget(btn_csv)
        btn_anim = QPushButton("导出3D动画")
        btn_anim.clicked.connect(self.export_animation)
        row2.addWidget(btn_anim)
        outer.addLayout(row2)

        self.status_label = QLabel(
            "高精度流程: 冻结清晰首帧 -> 左图逐个点击 P01~P15 -> 系统给右图建议点 -> 在右图修正 -> 保存当前帧 -> 自动跟踪下一帧并复核。"
        )
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)
        layout.addWidget(group)

    def _create_view_section(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        self.viewL = ClickableImageLabel("左图: 设置/修正当前点")
        self.viewL.setMinimumSize(640, 480)
        self.viewL.setStyleSheet("border:1px solid gray; background:black;")
        self.viewL.clicked.connect(self.on_click_left)
        row.addWidget(self.viewL, 1)
        self.viewR = ClickableImageLabel("右图: 修正当前点")
        self.viewR.setMinimumSize(640, 480)
        self.viewR.setStyleSheet("border:1px solid gray; background:black;")
        self.viewR.clicked.connect(self.on_click_right)
        row.addWidget(self.viewR, 1)
        layout.addLayout(row)

    def _create_log_section(self, layout: QVBoxLayout):
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(240)
        layout.addWidget(self.log)

    def log_add(self, text: str):
        self.log.append(text)

    def pick_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择YAML", YAML_DIR, "YAML (*.yaml *.yml)")
        if path:
            self.yaml_edit.setText(path)

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", RECORD_DIR, "Video (*.avi *.mp4 *.mkv)")
        if path:
            self.video_edit.setText(path)

    def load_yaml(self):
        yaml_path = self.yaml_edit.text().strip()
        if not yaml_path or not os.path.isfile(yaml_path):
            QMessageBox.warning(self, "错误", "请选择有效 YAML")
            return
        try:
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
            fs.release()
            self.proj_left = p1
            self.proj_right = p2
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(k1, d1, r1, p1, (self.image_width, self.image_height), cv2.CV_16SC2)
            self.map2x, self.map2y = cv2.initUndistortRectifyMap(k2, d2, r2, p2, (self.image_width, self.image_height), cv2.CV_16SC2)
            self.log_add(f"[OK] YAML 加载完成. 图像尺寸=({self.image_width}x{self.image_height}), 已准备高精度三角化参数。")
            QMessageBox.information(self, "完成", "YAML 加载完成。")
        except Exception as exc:
            self.log_add(f"[ERR] YAML 加载失败: {exc}")
            QMessageBox.warning(self, "错误", f"YAML 加载失败: {exc}")

    def open_video(self):
        video_path = self.video_edit.text().strip()
        if not video_path or not os.path.isfile(video_path):
            QMessageBox.warning(self, "错误", "请选择有效视频")
            return
        if self.map1x is None or self.proj_left is None or self.proj_right is None:
            QMessageBox.warning(self, "错误", "请先加载 YAML")
            return
        self._close_video_capture()
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            self.video_capture = None
            QMessageBox.warning(self, "错误", "无法打开视频")
            return
        self.video_fps = float(self.video_capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_index = -1
        self.is_playing = False
        self.btn_play.setText("播放")
        self.reset_tracking_session(keep_frame=True)
        self.log_add(f"[OK] 视频打开成功. FPS={self.video_fps:.3f}")
        self._read_and_show_one_frame()

    def _apply_display_fps(self):
        fps = int(self.disp_fps.value())
        self.play_timer.setInterval(max(1, int(round(1000.0 / float(fps)))))
        if self.is_playing:
            self.play_timer.start()

    def toggle_play(self):
        if self.video_capture is None:
            QMessageBox.warning(self, "错误", "请先打开视频")
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("暂停")
            self.play_timer.start()
        else:
            self.btn_play.setText("播放")
            self.play_timer.stop()

    def step_one(self):
        if self.video_capture is None:
            QMessageBox.warning(self, "错误", "请先打开视频")
            return
        if self.is_playing:
            self.is_playing = False
            self.btn_play.setText("播放")
            self.play_timer.stop()
        self._read_and_show_one_frame()

    def freeze_current_frame(self):
        if self.left_rectified is None or self.right_rectified is None:
            QMessageBox.warning(self, "错误", "当前没有可用帧")
            return
        if self.is_playing:
            self.is_playing = False
            self.btn_play.setText("播放")
            self.play_timer.stop()
        self.log_add(f"[OK] 已冻结当前帧 frame={self.frame_index}，可以开始或修正关键点。")
        self._refresh_views()

    def _on_play_tick(self):
        if self.is_playing and not self._read_and_show_one_frame():
            self.is_playing = False
            self.btn_play.setText("播放")
            self.play_timer.stop()
            self.log_add("[INFO] 视频已播放到末尾。")

    def _read_and_show_one_frame(self) -> bool:
        if self.video_capture is None:
            return False
        ret, frame = self.video_capture.read()
        if not ret or frame is None:
            return False
        self.current_frame = frame
        self.frame_index = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self._rectify_current_frame()
        self._refresh_views()
        return True

    def _rectify_current_frame(self):
        left, right = split_sbs(self.current_frame, SPLIT_OFFSET, SPLIT_GAP)
        self.left_rectified = cv2.remap(left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        self.right_rectified = cv2.remap(right, self.map2x, self.map2y, cv2.INTER_LINEAR)

    def _close_video_capture(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
        self.is_playing = False
        if self.video_capture is not None:
            try:
                self.video_capture.release()
            except Exception:
                pass
        self.video_capture = None

    def reset_tracking_session(self, keep_frame: bool = False):
        self.saved_frames = {}
        self.previous_saved_points = None
        self.previous_left_gray = None
        self.initialized = False
        self._reset_current_points()
        if not keep_frame:
            self.left_rectified = None
            self.right_rectified = None
            self.latest_left = None
            self.latest_right = None
        self.log_add("[INFO] 已重置追踪会话。")
        self._refresh_views()

    def _reset_current_points(self):
        self.current_points = [PointObservation(point_id=i + 1) for i in range(self.point_count)]
        self.selected_point_index = 0
        if hasattr(self, "point_index_spin"):
            self.point_index_spin.blockSignals(True)
            self.point_index_spin.setRange(1, self.point_count)
            self.point_index_spin.setValue(1)
            self.point_index_spin.blockSignals(False)

    def on_point_count_changed(self, value: int):
        self.point_count = int(value)
        self._reset_current_points()
        self._refresh_views()
        self.log_add(f"[INFO] 关键点数已设置为 {self.point_count}。")

    def on_selected_point_changed(self, value: int):
        self.selected_point_index = int(value) - 1
        self._refresh_views()

    def _shift_selected_point(self, delta: int):
        new_index = min(max(self.selected_point_index + delta, 0), self.point_count - 1)
        self.selected_point_index = new_index
        self.point_index_spin.blockSignals(True)
        self.point_index_spin.setValue(new_index + 1)
        self.point_index_spin.blockSignals(False)
        self._refresh_views()

    def start_manual_init(self):
        if self.left_rectified is None or self.right_rectified is None:
            QMessageBox.warning(self, "错误", "请先冻结一帧清晰画面")
            return
        self.saved_frames = {}
        self.previous_saved_points = None
        self.previous_left_gray = None
        self.initialized = False
        self._reset_current_points()
        self.log_add("[OK] 已进入首帧初始化。请依次在左图点击各点，再到右图精修对应点。")
        self._refresh_views()

    def clear_selected_point(self):
        obs = self.current_points[self.selected_point_index]
        obs.left_xy = None
        obs.right_xy = None
        obs.xyz = None
        obs.status = "pending"
        obs.confidence = 0.0
        self.log_add(f"[INFO] 已清除 P{obs.point_id:02d}。")
        self._refresh_views()

    def clear_current_points(self):
        self._reset_current_points()
        self.log_add("[INFO] 已清空当前帧的全部关键点。")
        self._refresh_views()
    def on_click_left(self, u: int, v: int):
        if self.left_rectified is None or self.right_rectified is None:
            self.log_add("[WARN] 当前没有可点击的帧。")
            return
        if self.is_playing:
            self.log_add("[HINT] 先暂停或冻结当前帧再修正点。")
            return
        obs = self.current_points[self.selected_point_index]
        obs.left_xy = (float(u), float(v))
        obs.status = "manual_left"
        match = self._match_right_point(float(u), float(v), self._previous_disparity(obs.point_id))
        if match is not None:
            obs.right_xy, obs.confidence = match
            obs.status = "manual_left+auto_right"
        self._update_point_3d(obs)
        self.log_add(self._point_summary(obs, prefix="[PT] "))
        if obs.right_xy is not None and self.selected_point_index < self.point_count - 1:
            self._shift_selected_point(1)
        else:
            self._refresh_views()

    def on_click_right(self, u: int, v: int):
        if self.right_rectified is None:
            self.log_add("[WARN] 当前没有可点击的帧。")
            return
        if self.is_playing:
            self.log_add("[HINT] 先暂停或冻结当前帧再修正点。")
            return
        obs = self.current_points[self.selected_point_index]
        obs.right_xy = (float(u), float(v))
        if obs.left_xy is not None:
            obs.status = "manual_pair" if obs.status == "manual_left" else "manual_corrected"
        self._update_point_3d(obs)
        self.log_add(self._point_summary(obs, prefix="[PT] "))
        if self.selected_point_index < self.point_count - 1:
            self._shift_selected_point(1)
        else:
            self._refresh_views()

    def _previous_disparity(self, point_id: int) -> Optional[float]:
        if not self.previous_saved_points:
            return None
        prev = self.previous_saved_points[point_id - 1]
        if prev.left_xy is None or prev.right_xy is None:
            return None
        return float(prev.left_xy[0] - prev.right_xy[0])

    def _match_right_point(self, u_left: float, v_left: float, prev_disparity: Optional[float], search_half_width: int = 40, row_half_height: int = 3, patch_half_size: int = 8) -> Optional[Tuple[Tuple[float, float], float]]:
        if self.left_rectified is None or self.right_rectified is None:
            return None
        left_gray = cv2.cvtColor(self.left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.right_rectified, cv2.COLOR_BGR2GRAY)
        h, w = left_gray.shape[:2]
        u = int(round(u_left))
        v = int(round(v_left))
        x0, x1 = u - patch_half_size, u + patch_half_size + 1
        y0, y1 = v - patch_half_size, v + patch_half_size + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        template = left_gray[y0:y1, x0:x1]
        if prev_disparity is None:
            center_x = max(patch_half_size, min(w - patch_half_size - 1, u - 60))
            min_x = max(patch_half_size, 0)
            max_x = min(u - patch_half_size - 1, w - patch_half_size - 1)
        else:
            center_x = int(round(u_left - prev_disparity))
            min_x = max(patch_half_size, center_x - search_half_width)
            max_x = min(u - patch_half_size - 1, center_x + search_half_width)
        if max_x <= min_x:
            return None
        min_y = max(patch_half_size, v - row_half_height)
        max_y = min(h - patch_half_size - 1, v + row_half_height)
        if max_y <= min_y:
            return None
        search = right_gray[min_y - patch_half_size:max_y + patch_half_size + 1, min_x - patch_half_size:max_x + patch_half_size + 1]
        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            return None
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        match_x = min_x + max_loc[0]
        match_y = min_y + max_loc[1]
        return (float(match_x), float(match_y)), float(max_val)

    def _update_point_3d(self, obs: PointObservation):
        if obs.left_xy is None or obs.right_xy is None or self.proj_left is None or self.proj_right is None:
            obs.xyz = None
            self._refresh_views()
            return
        u_l, v_l = obs.left_xy
        u_r, v_r = obs.right_xy
        disparity = u_l - u_r
        if disparity <= 0.1:
            obs.xyz = None
            self.log_add(f"[WARN] P{obs.point_id:02d} disparity={disparity:.3f}px，请修正右图点。")
            self._refresh_views()
            return
        pts_l = np.array([[u_l], [v_l]], dtype=np.float64)
        pts_r = np.array([[u_r], [v_r]], dtype=np.float64)
        homog = cv2.triangulatePoints(self.proj_left, self.proj_right, pts_l, pts_r)
        w = homog[3, 0]
        obs.xyz = None if abs(w) < 1e-12 else (homog[:3, 0] / w).astype(np.float64)
        self._refresh_views()

    def _point_summary(self, obs: PointObservation, prefix: str = "") -> str:
        left_text = "-" if obs.left_xy is None else f"({obs.left_xy[0]:.1f}, {obs.left_xy[1]:.1f})"
        right_text = "-" if obs.right_xy is None else f"({obs.right_xy[0]:.1f}, {obs.right_xy[1]:.1f})"
        xyz_text = "-" if obs.xyz is None else f"({obs.xyz[0]:.2f}, {obs.xyz[1]:.2f}, {obs.xyz[2]:.2f}) mm"
        return f"{prefix}P{obs.point_id:02d} L={left_text} R={right_text} 3D={xyz_text} status={obs.status} conf={obs.confidence:.3f}"

    def save_current_frame(self):
        if self.left_rectified is None or self.right_rectified is None:
            QMessageBox.warning(self, "错误", "当前没有可保存的帧")
            return
        missing = [obs.point_id for obs in self.current_points if obs.xyz is None]
        if missing:
            QMessageBox.warning(self, "错误", f"这些点还未完成三维重建: {missing}")
            return
        snapshot = deepcopy(self.current_points)
        self.saved_frames[self.frame_index] = snapshot
        self.previous_saved_points = deepcopy(snapshot)
        self.previous_left_gray = cv2.cvtColor(self.left_rectified, cv2.COLOR_BGR2GRAY)
        self.initialized = True
        v_errors = [abs(obs.left_xy[1] - obs.right_xy[1]) for obs in snapshot if obs.left_xy is not None and obs.right_xy is not None]
        mean_v_error = float(np.mean(v_errors)) if v_errors else 0.0
        self.log_add(f"[OK] 已保存 frame={self.frame_index} 的 {len(snapshot)} 个点。平均 |vL-vR| = {mean_v_error:.3f}px。")
        if mean_v_error > 1.5:
            self.log_add("[WARN] 当前帧上下行误差偏大，建议检查右图修正点或标定质量。")

    def track_next_frame(self):
        if not self.initialized or self.previous_left_gray is None or self.previous_saved_points is None:
            QMessageBox.warning(self, "错误", "请先完成首帧初始化并保存当前帧")
            return
        if self.video_capture is None:
            QMessageBox.warning(self, "错误", "请先打开视频")
            return
        if self.is_playing:
            self.is_playing = False
            self.btn_play.setText("播放")
            self.play_timer.stop()
        if not self._read_and_show_one_frame():
            QMessageBox.information(self, "结束", "视频已经到末尾")
            return
        current_gray = cv2.cvtColor(self.left_rectified, cv2.COLOR_BGR2GRAY)
        prev_pts = np.array([obs.left_xy for obs in self.previous_saved_points], dtype=np.float32).reshape(-1, 1, 2)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.previous_left_gray,
            current_gray,
            prev_pts,
            None,
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        new_points: List[PointObservation] = []
        for idx, prev_obs in enumerate(self.previous_saved_points):
            obs = PointObservation(point_id=prev_obs.point_id)
            if status is None or int(status[idx, 0]) == 0:
                obs.status = "lost"
                new_points.append(obs)
                continue
            x, y = next_pts[idx, 0]
            obs.left_xy = (float(x), float(y))
            lk_err = float(err[idx, 0]) if err is not None else 0.0
            lk_conf = 1.0 / (1.0 + max(lk_err, 0.0))
            prev_disp = None if prev_obs.left_xy is None or prev_obs.right_xy is None else float(prev_obs.left_xy[0] - prev_obs.right_xy[0])
            match = self._match_right_point(float(x), float(y), prev_disp)
            if match is not None:
                obs.right_xy, match_conf = match
                obs.confidence = min(lk_conf, match_conf)
                obs.status = "auto_tracked"
                self._update_point_3d(obs)
            else:
                obs.confidence = lk_conf
                obs.status = "need_right_fix"
            new_points.append(obs)
        self.current_points = new_points
        focus_idx = 0
        for idx, obs in enumerate(self.current_points):
            if obs.xyz is None or obs.confidence < 0.65:
                focus_idx = idx
                break
        self.selected_point_index = focus_idx
        self.point_index_spin.blockSignals(True)
        self.point_index_spin.setValue(focus_idx + 1)
        self.point_index_spin.blockSignals(False)
        ok_count = sum(1 for obs in self.current_points if obs.xyz is not None)
        self.log_add(f"[OK] 已自动跟踪到 frame={self.frame_index}，可用点 {ok_count}/{self.point_count}。请重点复核低置信度点。")
        self._refresh_views()
    def _refresh_views(self):
        if self.left_rectified is None or self.right_rectified is None:
            return
        left = self.left_rectified.copy()
        right = self.right_rectified.copy()
        self._draw_guides(left)
        self._draw_guides(right)
        for idx, obs in enumerate(self.current_points):
            selected = idx == self.selected_point_index
            self._draw_point(left, obs.left_xy, obs, selected, True)
            self._draw_point(right, obs.right_xy, obs, selected, False)
        self.latest_left = left
        self.latest_right = right
        self.viewL.set_image_pixmap(bgr_to_qpixmap(left), left.shape[1], left.shape[0])
        self.viewR.set_image_pixmap(bgr_to_qpixmap(right), right.shape[1], right.shape[0])

    def _draw_guides(self, img: np.ndarray):
        step = max(60, img.shape[0] // 10)
        for y in range(step, img.shape[0], step):
            cv2.line(img, (0, y), (img.shape[1] - 1, y), (0, 80, 0), 1)

    def _draw_point(self, img: np.ndarray, pt: Optional[Tuple[float, float]], obs: PointObservation, selected: bool, is_left: bool):
        if pt is None:
            return
        x = int(round(pt[0]))
        y = int(round(pt[1]))
        if obs.status in ("auto_tracked", "manual_left+auto_right"):
            color = (0, 255, 255)
        elif obs.status in ("manual_pair", "manual_corrected"):
            color = (0, 255, 0)
        elif obs.status in ("lost", "need_right_fix") or obs.xyz is None:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        size = 26 if selected else 18
        thickness = 3 if selected else 2
        cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=size, thickness=thickness)
        side = "L" if is_left else "R"
        cv2.putText(img, f"P{obs.point_id:02d}{side}", (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if selected:
            cv2.circle(img, (x, y), 16, (255, 255, 255), 1)

    def export_csv(self):
        if not self.saved_frames:
            QMessageBox.warning(self, "错误", "还没有保存任何帧")
            return
        default_name = os.path.join(RECORD_DIR, f"tracked_points_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        path, _ = QFileDialog.getSaveFileName(self, "导出CSV", default_name, "CSV (*.csv)")
        if not path:
            return
        rows = []
        for frame_idx in sorted(self.saved_frames.keys()):
            t_sec = frame_idx / self.video_fps if self.video_fps > 1e-9 else 0.0
            for obs in self.saved_frames[frame_idx]:
                lxy = obs.left_xy or (None, None)
                rxy = obs.right_xy or (None, None)
                xyz = obs.xyz.tolist() if obs.xyz is not None else [None, None, None]
                rows.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": f"{t_sec:.6f}",
                    "point_id": obs.point_id,
                    "u_left": lxy[0],
                    "v_left": lxy[1],
                    "u_right": rxy[0],
                    "v_right": rxy[1],
                    "x_mm": xyz[0],
                    "y_mm": xyz[1],
                    "z_mm": xyz[2],
                    "track_status": obs.status,
                    "confidence": f"{obs.confidence:.6f}",
                })
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_idx", "timestamp_sec", "point_id", "u_left", "v_left", "u_right", "v_right", "x_mm", "y_mm", "z_mm", "track_status", "confidence"])
            writer.writeheader()
            writer.writerows(rows)
        self.log_add(f"[OK] CSV 已导出到: {path}")
        QMessageBox.information(self, "完成", f"CSV 已导出到:\n{path}")

    def export_animation(self):
        if not self.saved_frames:
            QMessageBox.warning(self, "错误", "还没有保存任何帧")
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            QMessageBox.warning(self, "错误", f"导出3D动画需要 matplotlib: {exc}")
            return
        default_name = os.path.join(RECORD_DIR, f"tracked_3d_animation_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        path, _ = QFileDialog.getSaveFileName(self, "导出3D动画", default_name, "MP4 (*.mp4)")
        if not path:
            return
        frames = []
        for frame_idx in sorted(self.saved_frames.keys()):
            xyz = np.array([obs.xyz for obs in self.saved_frames[frame_idx] if obs.xyz is not None], dtype=np.float64)
            if len(xyz) == self.point_count:
                frames.append((frame_idx, xyz))
        if not frames:
            QMessageBox.warning(self, "错误", "没有完整的三维帧可导出")
            return
        all_xyz = np.concatenate([xyz for _, xyz in frames], axis=0)
        mins = np.min(all_xyz, axis=0)
        maxs = np.max(all_xyz, axis=0)
        center = 0.5 * (mins + maxs)
        radius = max(float(np.max(maxs - mins)) * 0.6, 50.0)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, min(self.video_fps, 30.0)), (960, 720))
        if not writer.isOpened():
            QMessageBox.warning(self, "错误", "无法创建动画文件")
            return
        try:
            fig = plt.figure(figsize=(9.6, 7.2), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            for frame_idx, xyz in frames:
                ax.clear()
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="tab:orange", s=40, depthshade=False)
                ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="tab:blue", linewidth=2)
                for point_idx, point in enumerate(xyz, start=1):
                    ax.text(point[0], point[1], point[2], f"P{point_idx:02d}", fontsize=8)
                ax.set_title(f"Tracked 3D Skeleton | frame={frame_idx}")
                ax.set_xlabel("X (mm)")
                ax.set_ylabel("Y (mm)")
                ax.set_zlabel("Z (mm)")
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
                ax.set_zlim(center[2] - radius, center[2] + radius)
                ax.view_init(elev=24, azim=-58)
                ax.grid(True)
                fig.tight_layout()
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
            plt.close("all")
        self.log_add(f"[OK] 3D 动画已导出到: {path}")
        QMessageBox.information(self, "完成", f"3D 动画已导出到:\n{path}")

    def closeEvent(self, event):
        self._close_video_capture()
        event.accept()
