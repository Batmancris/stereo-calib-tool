# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from typing import Optional, Tuple

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QTextEdit, QMessageBox, QDoubleSpinBox, QSpinBox
)
from PyQt5.QtGui import QImage, QPixmap

from utils_img import split_sbs
from utils_common import read_mat, bgr_to_qpixmap
from config import RECORD_DIR, YAML_DIR
from clickable_label import ClickableImageLabel


def reproject_point(u: float, v: float, d: float, reprojection_matrix: np.ndarray) -> np.ndarray:
    """
    使用重投影矩阵将像素坐标和视差转换为三维坐标
    
    参数：
        u: 像素点的 u 坐标（水平方向）
        v: 像素点的 v 坐标（垂直方向）
        d: 视差，即左右相机中对应点的水平像素差
        reprojection_matrix: 重投影矩阵，由 stereoRectify 函数计算得到
    
    返回：
        三维坐标数组 [X, Y, Z]，单位与标定板方格尺寸一致
    
    原理：
        使用公式：[X, Y, Z, W]^T = Q * [u, v, d, 1]^T
        然后通过透视除法：X = X/W, Y = Y/W, Z = Z/W
    """
    vec = reprojection_matrix @ np.array([float(u), float(v), float(d), 1.0], dtype=np.float64)
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

        # 左右各自的“显示图”(带辅助线和点)
        self.latest_left: Optional[np.ndarray] = None
        self.latest_right: Optional[np.ndarray] = None

        # 点：全部用“各自图像坐标”(u,v)，不再用拼接坐标
        self.point_A_left: Optional[Tuple[float, float]] = None
        self.point_A_right: Optional[Tuple[float, float]] = None
        self.point_B_left: Optional[Tuple[float, float]] = None
        self.point_B_right: Optional[Tuple[float, float]] = None

        self._init_ui()

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._on_play_tick)
        self._apply_display_fps()

    def _init_ui(self):
        # 创建滚动区域
        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # 主容器
        container = QWidget()
        lay = QVBoxLayout(container)

        # 添加各个部分
        self._create_yaml_section(lay)
        self._create_video_section(lay)
        self._create_measure_section(lay)
        self._create_view_section(lay)
        self._create_log_section(lay)

        # 设置滚动区域
        scroll_area.setWidget(container)
        main_lay = QVBoxLayout(self)
        main_lay.addWidget(scroll_area)

    def _create_yaml_section(self, layout: QVBoxLayout) -> None:
        """创建YAML配置部分"""
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
        layout.addLayout(r1)

    def _create_video_section(self, layout: QVBoxLayout) -> None:
        """创建视频控制部分"""
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

        layout.addLayout(r2)

    def _create_measure_section(self, layout: QVBoxLayout) -> None:
        """创建测量控制部分"""
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

        layout.addLayout(r3)

    def _create_view_section(self, layout: QVBoxLayout) -> None:
        """创建视图显示部分"""
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

        layout.addLayout(view_row)

    def _create_log_section(self, layout: QVBoxLayout) -> None:
        """创建日志显示部分"""
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(220)
        layout.addWidget(self.log)

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

    def load_yaml(self) -> None:
        """加载YAML文件并预计算矫正映射"""
        yp = self.yaml_edit.text().strip()
        if not yp or not os.path.isfile(yp):
            QMessageBox.warning(self, "错误", "请选择有效 YAML")
            return

        fs = cv2.FileStorage(yp, cv2.FILE_STORAGE_READ)
        self.image_width = int(fs.getNode("image_width").real())
        self.image_height = int(fs.getNode("image_height").real())
        K1 = read_mat(fs, "K1"); D1 = read_mat(fs, "D1")
        K2 = read_mat(fs, "K2"); D2 = read_mat(fs, "D2")
        R1 = read_mat(fs, "R1"); P1 = read_mat(fs, "P1")
        R2 = read_mat(fs, "R2"); P2 = read_mat(fs, "P2")
        self.reprojection_matrix = read_mat(fs, "Q")
        fs.release()

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (self.image_width, self.image_height), cv2.CV_16SC2)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (self.image_width, self.image_height), cv2.CV_16SC2)

        self.log_add(f"[OK] YAML loaded. image=({self.image_width}x{self.image_height}), rect maps ready.")
        QMessageBox.information(self, "完成", "YAML 加载完成，矫正映射已预计算。")

    def open_video(self) -> None:
        """打开视频文件"""
        vp = self.video_edit.text().strip()
        if not vp or not os.path.isfile(vp):
            QMessageBox.warning(self, "错误", "请选择有效视频")
            return
        if self.map1x is None or self.reprojection_matrix is None:
            QMessageBox.warning(self, "错误", "请先加载 YAML（预计算矫正映射）")
            return

        self._close_video_capture()
        self.video_capture = cv2.VideoCapture(vp)
        if not self.video_capture.isOpened():
            self.video_capture = None
            QMessageBox.warning(self, "错误", "无法打开视频")
            return

        self.is_playing = False
        self.btn_play.setText("▶ 播放")
        self.clear_points()

        self.log_add("[OK] video opened. 播放/步进到清晰帧后暂停，再点“冻结为测距帧”。")
        self._read_and_show_one_frame()

    def _apply_display_fps(self) -> None:
        """应用显示帧率设置"""
        fps = int(self.disp_fps.value())
        interval_ms = max(1, int(round(1000.0 / float(fps))))
        self.play_timer.setInterval(interval_ms)
        if self.is_playing:
            self.play_timer.start()

    def toggle_play(self) -> None:
        """切换播放状态"""
        if self.video_capture is None:
            QMessageBox.warning(self, "错误", "请先打开视频")
            return
        if self.map1x is None:
            QMessageBox.warning(self, "错误", "请先加载 YAML")
            return

        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("⏸ 暂停")
            self.play_timer.start()
        else:
            self.btn_play.setText("▶ 播放")
            self.play_timer.stop()

    def step_one(self) -> None:
        """步进一帧"""
        if self.video_capture is None:
            QMessageBox.warning(self, "错误", "请先打开视频")
            return
        if self.is_playing:
            self.is_playing = False
            self.btn_play.setText("▶ 播放")
            self.play_timer.stop()
        self._read_and_show_one_frame()

    def freeze_for_measure(self) -> None:
        """冻结当前帧为测距帧"""
        if self.left_rectified is None or self.right_rectified is None:
            QMessageBox.warning(self, "错误", "当前没有可用帧（请先播放/步进）")
            return

        if self.is_playing:
            self.is_playing = False
            self.btn_play.setText("▶ 播放")
            self.play_timer.stop()

        self.clear_points()
        self.log_add("[OK] 已冻结当前帧为测距帧。请按 A左→A右→B左→B右 点击取点。")
        self._update_visualization()
        self._show_latest_frames()

    def _on_play_tick(self) -> None:
        """播放定时器回调"""
        if not self.is_playing:
            return
        ok = self._read_and_show_one_frame()
        if not ok:
            self.is_playing = False
            self.btn_play.setText("▶ 播放")
            self.play_timer.stop()
            self.log_add("[INFO] 视频到末尾，已停止。")

    def _read_and_show_one_frame(self) -> bool:
        """读取并显示一帧"""
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
        """矫正当前帧"""
        left, right = split_sbs(self.current_frame, 0, 0)
        self.left_rectified = cv2.remap(left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        self.right_rectified = cv2.remap(right, self.map2x, self.map2y, cv2.INTER_LINEAR)

    def _close_video_capture(self) -> None:
        """关闭视频捕获"""
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
        """清除所有点"""
        self.point_A_left = self.point_A_right = self.point_B_left = self.point_B_right = None
        self.log_add("[INFO] cleared points. 点击顺序：A左→A右→B左→B右")

    def _update_visualization(self) -> None:
        """更新可视化"""
        if self.left_rectified is None or self.right_rectified is None:
            return

        L = self.left_rectified.copy()
        R = self.right_rectified.copy()

        # 画水平线（用于检验矫正后同名点应同一水平线）
        step = max(1, self.image_height // 10)
        for y in range(step, self.image_height, step):
            cv2.line(L, (0, y), (L.shape[1] - 1, y), (0, 255, 0), 1)
            cv2.line(R, (0, y), (R.shape[1] - 1, y), (0, 255, 0), 1)

        def draw_pt(img: np.ndarray, pt: Optional[Tuple[float, float]], color: Tuple[int, int, int], label: str) -> None:
            """绘制点"""
            if pt is None:
                return
            x, y = int(pt[0]), int(pt[1])
            cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(img, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        draw_pt(L, self.point_A_left, (0, 0, 255), "A_L")
        draw_pt(R, self.point_A_right, (0, 0, 255), "A_R")
        draw_pt(L, self.point_B_left, (255, 0, 0), "B_L")
        draw_pt(R, self.point_B_right, (255, 0, 0), "B_R")

        self.latest_left = L
        self.latest_right = R

    def _show_latest_frames(self) -> None:
        """显示最新帧"""
        if self.latest_left is not None:
            pmL = bgr_to_qpixmap(self.latest_left)
            self.viewL.set_image_pixmap(pmL, self.latest_left.shape[1], self.latest_left.shape[0])
        if self.latest_right is not None:
            pmR = bgr_to_qpixmap(self.latest_right)
            self.viewR.set_image_pixmap(pmR, self.latest_right.shape[1], self.latest_right.shape[0])

    # ---------- 点击逻辑：左右分别处理 ----------
    def on_click_left(self, u: float, v: float) -> None:
        """左图点击事件"""
        if self.left_rectified is None:
            self.log_add("[WARN] 还没有帧可点")
            return
        if self.is_playing:
            self.log_add("[HINT] 建议先暂停并点“冻结为测距帧”，再取点。")
            return

        if self.point_A_left is None:
            self.point_A_left = (u, v)
            self.log_add(f"[PT] A_L = {(u, v)}")
        elif self.point_B_left is None:
            self.point_B_left = (u, v)
            self.log_add(f"[PT] B_L = {(u, v)}")
        else:
            self.log_add("[INFO] 左图点已选完（A_L、B_L）。")
        self._update_visualization()
        self._show_latest_frames()

    def on_click_right(self, u: float, v: float) -> None:
        """右图点击事件"""
        if self.right_rectified is None:
            self.log_add("[WARN] 还没有帧可点")
            return
        if self.is_playing:
            self.log_add("[HINT] 建议先暂停并点“冻结为测距帧”，再取点。")
            return

        if self.point_A_left is None:
            self.log_add("[HINT] 第1步请先在左图点 A_L")
            return

        if self.point_A_right is None:
            self.point_A_right = (u, v)
            self.log_add(f"[PT] A_R = {(u, v)}")
        elif self.point_B_left is None:
            self.log_add("[HINT] 第3步请先在左图点 B_L")
            return
        elif self.point_B_right is None:
            self.point_B_right = (u, v)
            self.log_add(f"[PT] B_R = {(u, v)}")
        else:
            self.log_add("[INFO] 右图点已选完（A_R、B_R）。")
        self._update_visualization()
        self._show_latest_frames()

    def compute_distance(self) -> None:
        """
        计算两点之间的三维距离
        
        步骤：
            1. 验证必要的参数是否已设置
            2. 获取四个点的像素坐标
            3. 检查左右相机中对应点的垂直坐标是否一致（验证矫正效果）
            4. 计算视差
            5. 使用重投影矩阵将像素坐标和视差转换为三维坐标
            6. 计算两点之间的欧氏距离
            7. 与已知长度比较，计算误差
            8. 显示结果
        """
        if self.reprojection_matrix is None or self.image_width is None:
            QMessageBox.warning(self, "错误", "请先加载 YAML 并打开视频/冻结帧")
            return
        if any(x is None for x in [self.point_A_left, self.point_A_right, self.point_B_left, self.point_B_right]):
            QMessageBox.warning(self, "错误", "请先按顺序点击四个点：A左→A右→B左→B右")
            return

        # 获取四个点的像素坐标
        uL_A, vL_A = float(self.point_A_left[0]), float(self.point_A_left[1])
        uR_A, vR_A = float(self.point_A_right[0]), float(self.point_A_right[1])
        uL_B, vL_B = float(self.point_B_left[0]), float(self.point_B_left[1])
        uR_B, vR_B = float(self.point_B_right[0]), float(self.point_B_right[1])

        def v_check(name: str, vL: float, vR: float) -> None:
            """检查左右相机中对应点的垂直坐标是否一致"""
            dv = abs(vL - vR)
            if dv > 2.0:
                self.log_add(f"[WARN] {name}: |vL-vR|={dv:.2f}px 偏大：要么点错对应点，要么矫正不够好。")

        # 检查垂直坐标一致性（验证矫正效果）
        v_check("A", vL_A, vR_A)
        v_check("B", vL_B, vR_B)

        # 计算平均垂直坐标
        vA = 0.5 * (vL_A + vR_A)
        vB = 0.5 * (vL_B + vR_B)

        # 计算视差
        dA = uL_A - uR_A
        dB = uL_B - uR_B
        if dA <= 0 or dB <= 0:
            self.log_add(f"[ERR] disparity<=0: dA={dA:.3f}, dB={dB:.3f}（通常是对应点点错/左右反了）")
            return

        try:
            # 重投影到三维空间
            PA = reproject_point(uL_A, vA, dA, self.reprojection_matrix)
            PB = reproject_point(uL_B, vB, dB, self.reprojection_matrix)
        except Exception as e:
            self.log_add(f"[ERR] reprojection failed: {e}")
            return

        # 计算三维距离
        dist = float(np.linalg.norm(PA - PB))
        gt = float(self.len_mm.value())
        err = dist - gt
        rel = (err / gt) * 100.0 if gt > 1e-9 else 0.0

        # 记录和显示结果
        self.log_add(f"[3D] A(mm) = {PA.round(2)}   dA={dA:.2f}px")
        self.log_add(f"[3D] B(mm) = {PB.round(2)}   dB={dB:.2f}px")
        self.log_add(f"[MEAS] distance = {dist:.2f} mm | GT = {gt:.2f} mm | error = {err:+.2f} mm ({rel:+.2f}%)")

        QMessageBox.information(
            self, "测距结果",
            f"测得距离: {dist:.2f} mm\n真实长度: {gt:.2f} mm\n误差: {err:+.2f} mm ({rel:+.2f}%)"
        )
