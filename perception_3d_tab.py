# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from typing import Optional, Tuple, List
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QTextEdit, QMessageBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QGroupBox
)
from PyQt5.QtGui import QImage, QPixmap

from utils_img import split_sbs
from utils_common import read_mat, bgr_to_qpixmap
from config import RECORD_DIR, YAML_DIR, PLY_DIR


def disparity_to_color(disp: np.ndarray) -> np.ndarray:
    """
    将视差图转换为彩色图像，便于可视化
    """
    if disp is None:
        return None
    # 归一化到0-255
    disp_normalized = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    disp_normalized = np.uint8(disp_normalized)
    # 应用伪彩色
    disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
    return disp_color


class DisparityThread(QThread):
    """
    视差计算线程
    """
    disparity_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)

    def __init__(self, left, right, method='SGBM', min_disparity=0, num_disparities=16, block_size=9):
        super().__init__()
        self.left = left
        self.right = right
        self.method = method
        self.min_disparity = min_disparity
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.running = True

    def run(self):
        try:
            # 转换为灰度图
            left_gray = cv2.cvtColor(self.left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(self.right, cv2.COLOR_BGR2GRAY)

            if self.method == 'SGBM':
                # 使用SGBM算法计算视差
                stereo = cv2.StereoSGBM_create(
                    minDisparity=self.min_disparity,
                    numDisparities=self.num_disparities,
                    blockSize=self.block_size,
                    P1=8 * 3 * self.block_size ** 2,
                    P2=32 * 3 * self.block_size ** 2,
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                )
            else:
                # 使用BM算法计算视差
                stereo = cv2.StereoBM_create(
                    numDisparities=self.num_disparities,
                    blockSize=self.block_size
                )
                stereo.setMinDisparity(self.min_disparity)

            # 计算视差
            disp = stereo.compute(left_gray, right_gray)

            # 处理视差图
            if self.method == 'SGBM':
                # SGBM返回的视差需要除以16
                disp = disp.astype(np.float32) / 16.0

            if self.running:
                self.disparity_signal.emit(disp)
                self.log_signal.emit(f"[OK] 视差计算完成，方法: {self.method}, 视差范围: [{disp.min():.2f}, {disp.max():.2f}]")
        except Exception as e:
            self.log_signal.emit(f"[ERR] 视差计算失败: {str(e)}")

    def stop(self):
        self.running = False


class PointCloudGenerator:
    """
    三维点云生成器
    """
    def __init__(self, reprojection_matrix: np.ndarray):
        """
        初始化点云生成器
        
        参数:
            reprojection_matrix: 重投影矩阵，由stereoRectify函数计算得到
        """
        self.reprojection_matrix = reprojection_matrix

    def generate(self, left_rect: np.ndarray, disparity_map: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从矫正后的左图和视差图生成三维点云
        
        参数:
            left_rect: 矫正后的左图
            disparity_map: 视差图
            mask: 掩码，用于过滤点云
        
        返回:
            三维点云 (N, 3)，单位与标定板方格尺寸一致
            点云颜色 (N, 3)，RGB格式
        """
        if left_rect is None or disparity_map is None:
            return None, None

        h, w = left_rect.shape[:2]
        points = []
        colors = []

        # 遍历图像像素
        for v in range(0, h, 4):  # 每隔4个像素采样，减少点云数量
            for u in range(0, w, 4):
                disparity = disparity_map[v, u]
                if disparity <= 5:  # 提高视差阈值，过滤掉过小的视差值（噪声）
                    continue

                # 重投影到三维空间
                vec = self.reprojection_matrix @ np.array([float(u), float(v), float(disparity), 1.0], dtype=np.float64)
                W = vec[3]
                if abs(W) < 1e-12:
                    continue

                X = vec[0] / W
                Y = vec[1] / W
                Z = vec[2] / W

                # 过滤过远或过近的点
                if Z < 100 or Z > 3000:  # 调整为更合理的距离范围：100mm到3000mm
                    continue

                # 收集点和颜色
                points.append([X, Y, Z])
                colors.append(left_rect[v, u][::-1])  # BGR转RGB

        if len(points) == 0:
            return None, None

        return np.array(points), np.array(colors)


class Perception3DTab(QWidget):
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
        self.disparity_map: Optional[np.ndarray] = None
        self.point_cloud: Optional[np.ndarray] = None
        self.point_colors: Optional[np.ndarray] = None

        self.disparity_thread: Optional[DisparityThread] = None

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
        self._create_parameter_section(lay)
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

        bload = QPushButton("加载YAML")
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

        b_freeze = QPushButton("冻结为处理帧")
        b_freeze.clicked.connect(self.freeze_for_process)
        r2.addWidget(b_freeze)

        layout.addLayout(r2)

    def _create_parameter_section(self, layout: QVBoxLayout) -> None:
        """创建视差计算参数部分"""
        param_group = QGroupBox("视差计算参数")
        param_lay = QVBoxLayout()

        # 参数说明
        info_label = QLabel("参数选择建议：")
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setFixedHeight(100)
        info_text.setPlainText("纹理丰富场景：块大小 5-7，视差数量 32-64\n" +
                              "纹理较少场景：块大小 11-15，视差数量 64-128\n" +
                              "近距离场景：最小视差 0-10，视差数量适中\n" +
                              "远距离场景：最小视差 10-20，视差数量较大")
        param_lay.addWidget(info_label)
        param_lay.addWidget(info_text)

        # 参数控件
        control_lay = QHBoxLayout()
        control_lay.addWidget(QLabel("方法:"))
        self.disp_method = QComboBox()
        self.disp_method.addItems(["SGBM", "BM"])
        control_lay.addWidget(self.disp_method)

        control_lay.addWidget(QLabel("最小视差:"))
        self.min_disparity = QSpinBox()
        self.min_disparity.setRange(0, 100)
        self.min_disparity.setValue(0)
        control_lay.addWidget(self.min_disparity)

        control_lay.addWidget(QLabel("视差数量:"))
        self.num_disparities = QSpinBox()
        self.num_disparities.setRange(16, 256)
        self.num_disparities.setSingleStep(16)
        self.num_disparities.setValue(16)
        control_lay.addWidget(self.num_disparities)

        control_lay.addWidget(QLabel("块大小:"))
        self.block_size = QSpinBox()
        self.block_size.setRange(5, 25)
        self.block_size.setSingleStep(2)
        self.block_size.setValue(9)
        control_lay.addWidget(self.block_size)

        param_lay.addLayout(control_lay)

        # 预设按钮
        preset_lay = QHBoxLayout()
        btn_preset1 = QPushButton("纹理丰富场景")
        btn_preset1.clicked.connect(lambda: self.apply_preset(0))
        preset_lay.addWidget(btn_preset1)

        btn_preset2 = QPushButton("纹理较少场景")
        btn_preset2.clicked.connect(lambda: self.apply_preset(1))
        preset_lay.addWidget(btn_preset2)

        btn_preset3 = QPushButton("近距离场景")
        btn_preset3.clicked.connect(lambda: self.apply_preset(2))
        preset_lay.addWidget(btn_preset3)

        btn_preset4 = QPushButton("远距离场景")
        btn_preset4.clicked.connect(lambda: self.apply_preset(3))
        preset_lay.addWidget(btn_preset4)

        param_lay.addLayout(preset_lay)

        # 操作按钮
        action_lay = QHBoxLayout()
        btn_compute_disp = QPushButton("计算视差")
        btn_compute_disp.clicked.connect(self.compute_disparity)
        action_lay.addWidget(btn_compute_disp)

        btn_generate_pc = QPushButton("生成点云")
        btn_generate_pc.clicked.connect(self.generate_point_cloud)
        action_lay.addWidget(btn_generate_pc)

        btn_view_pc = QPushButton("查看点云")
        btn_view_pc.clicked.connect(self.view_point_cloud)
        action_lay.addWidget(btn_view_pc)

        param_lay.addLayout(action_lay)

        param_group.setLayout(param_lay)
        layout.addWidget(param_group)

    def _create_view_section(self, layout: QVBoxLayout) -> None:
        """创建视图显示部分"""
        view_row = QHBoxLayout()

        # 左图
        self.viewL = QLabel("左图")
        self.viewL.setMinimumSize(420, 480)
        self.viewL.setStyleSheet("border:1px solid gray; background:black;")
        view_row.addWidget(self.viewL, 1)

        # 右图
        self.viewR = QLabel("右图")
        self.viewR.setMinimumSize(420, 480)
        self.viewR.setStyleSheet("border:1px solid gray; background:black;")
        view_row.addWidget(self.viewR, 1)

        # 视差图
        self.viewD = QLabel("视差图")
        self.viewD.setMinimumSize(420, 480)
        self.viewD.setStyleSheet("border:1px solid gray; background:black;")
        view_row.addWidget(self.viewD, 1)

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

    def load_yaml(self):
        yaml_path = self.yaml_edit.text().strip()
        if not yaml_path or not os.path.isfile(yaml_path):
            QMessageBox.warning(self, "错误", "请选择有效 YAML")
            return

        try:
            fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
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

            self.log_add(f"[OK] YAML 加载完成. 图像尺寸=({self.image_width}x{self.image_height}), 矫正映射已预计算.")
            QMessageBox.information(self, "完成", "YAML 加载完成，矫正映射已预计算。")
        except Exception as e:
            self.log_add(f"[ERR] 加载 YAML 失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"加载 YAML 失败: {str(e)}")

    def open_video(self):
        video_path = self.video_edit.text().strip()
        if not video_path or not os.path.isfile(video_path):
            QMessageBox.warning(self, "错误", "请选择有效视频")
            return
        if self.map1x is None or self.reprojection_matrix is None:
            QMessageBox.warning(self, "错误", "请先加载 YAML（预计算矫正映射）")
            return

        self._close_video_capture()
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            self.video_capture = None
            QMessageBox.warning(self, "错误", "无法打开视频")
            return

        self.is_playing = False
        self.btn_play.setText("▶ 播放")

        self.log_add("[OK] 视频打开成功。")
        self._read_and_show_one_frame()

    def _apply_display_fps(self):
        fps = 30
        interval_ms = max(1, int(round(1000.0 / float(fps))))
        self.play_timer.setInterval(interval_ms)
        if self.is_playing:
            self.play_timer.start()

    def toggle_play(self):
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

    def step_one(self):
        if self.video_capture is None:
            QMessageBox.warning(self, "错误", "请先打开视频")
            return
        if self.is_playing:
            self.is_playing = False
            self.btn_play.setText("▶ 播放")
            self.play_timer.stop()
        self._read_and_show_one_frame()

    def freeze_for_process(self):
        if self.left_rectified is None or self.right_rectified is None:
            QMessageBox.warning(self, "错误", "当前没有可用帧（请先播放/步进）")
            return

        if self.is_playing:
            self.is_playing = False
            self.btn_play.setText("▶ 播放")
            self.play_timer.stop()

        self.log_add("[OK] 已冻结当前帧为处理帧。")
        self._show_latest_lr()

    def _on_play_tick(self):
        if not self.is_playing:
            return
        ok = self._read_and_show_one_frame()
        if not ok:
            self.is_playing = False
            self.btn_play.setText("▶ 播放")
            self.play_timer.stop()
            self.log_add("[INFO] 视频到末尾，已停止。")

    def _read_and_show_one_frame(self) -> bool:
        if self.video_capture is None:
            return False
        ret, frame = self.video_capture.read()
        if not ret or frame is None:
            return False

        self.current_frame = frame
        self._rectify_current_frame()
        self._show_latest_lr()
        return True

    def _rectify_current_frame(self):
        left, right = split_sbs(self.current_frame, 0, 0)
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

    def _show_latest_lr(self):
        if self.left_rectified is not None:
            pmL = bgr_to_qpixmap(self.left_rectified)
            self.viewL.setPixmap(pmL.scaled(self.viewL.width(), self.viewL.height(), aspectRatioMode=1))
        if self.right_rectified is not None:
            pmR = bgr_to_qpixmap(self.right_rectified)
            self.viewR.setPixmap(pmR.scaled(self.viewR.width(), self.viewR.height(), aspectRatioMode=1))
        if self.disparity_map is not None:
            disp_color = disparity_to_color(self.disparity_map)
            if disp_color is not None:
                pmD = bgr_to_qpixmap(disp_color)
                self.viewD.setPixmap(pmD.scaled(self.viewD.width(), self.viewD.height(), aspectRatioMode=1))

    def compute_disparity(self):
        if self.left_rectified is None or self.right_rectified is None:
            QMessageBox.warning(self, "错误", "当前没有可用帧（请先播放/步进）")
            return

        # 停止之前的线程
        if self.disparity_thread is not None and self.disparity_thread.isRunning():
            self.disparity_thread.stop()
            self.disparity_thread.wait()

        # 启动新的视差计算线程
        method = self.disp_method.currentText()
        min_disparity = self.min_disparity.value()
        num_disparities = self.num_disparities.value()
        block_size = self.block_size.value()

        self.log_add(f"[INFO] 开始计算视差，方法: {method}, 最小视差: {min_disparity}, 视差数量: {num_disparities}, 块大小: {block_size}")

        self.disparity_thread = DisparityThread(
            self.left_rectified, self.right_rectified, method, min_disparity, num_disparities, block_size
        )
        self.disparity_thread.disparity_signal.connect(self.on_disparity_computed)
        self.disparity_thread.log_signal.connect(self.log_add)
        self.disparity_thread.start()

    def on_disparity_computed(self, disp):
        self.disparity_map = disp
        self._show_latest_lr()

    def generate_point_cloud(self):
        if self.reprojection_matrix is None:
            QMessageBox.warning(self, "错误", "请先加载 YAML")
            return
        if self.left_rectified is None:
            QMessageBox.warning(self, "错误", "当前没有可用帧")
            return
        if self.disparity_map is None:
            QMessageBox.warning(self, "错误", "请先计算视差")
            return

        try:
            # 生成点云
            generator = PointCloudGenerator(self.reprojection_matrix)
            points, colors = generator.generate(self.left_rectified, self.disparity_map)

            if points is None or len(points) == 0:
                self.log_add("[ERR] 点云生成失败：没有有效的点")
                QMessageBox.warning(self, "错误", "点云生成失败：没有有效的点")
                return

            # 点云后处理：过滤离群点
            if len(points) > 1000:
                self.log_add("[INFO] 进行点云后处理")
                
                # 计算点云的均值和标准差
                mean = np.mean(points, axis=0)
                std = np.std(points, axis=0)
                
                # 过滤离群点（距离均值超过3个标准差的点）
                dist = np.linalg.norm(points - mean, axis=1)
                mask = dist < 3 * np.max(std)
                filtered_points = points[mask]
                filtered_colors = colors[mask]
                
                if len(filtered_points) < len(points):
                    self.log_add(f"[INFO] 过滤后点云数量: {len(filtered_points)} (原: {len(points)})")
                    points = filtered_points
                    colors = filtered_colors

            self.point_cloud = points
            self.point_colors = colors

            # 保存点云为PLY格式
            self.save_point_cloud()

            self.log_add(f"[OK] 点云生成成功，点数: {len(points)}")
            QMessageBox.information(self, "完成", f"点云生成成功，点数: {len(points)}")
        except Exception as e:
            self.log_add(f"[ERR] 点云生成失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"点云生成失败: {str(e)}")

    def apply_preset(self, preset_id):
        """
        应用预设参数配置
        
        参数:
            preset_id: 预设ID
                0: 纹理丰富场景
                1: 纹理较少场景
                2: 近距离场景
                3: 远距离场景
        """
        presets = {
            0: {  # 纹理丰富场景
                'method': 'SGBM',
                'min_disparity': 0,
                'num_disparities': 64,
                'block_size': 7
            },
            1: {  # 纹理较少场景
                'method': 'SGBM',
                'min_disparity': 0,
                'num_disparities': 128,
                'block_size': 15
            },
            2: {  # 近距离场景
                'method': 'SGBM',
                'min_disparity': 0,
                'num_disparities': 64,
                'block_size': 9
            },
            3: {  # 远距离场景
                'method': 'SGBM',
                'min_disparity': 10,
                'num_disparities': 128,
                'block_size': 11
            }
        }

        if preset_id in presets:
            preset = presets[preset_id]
            self.disp_method.setCurrentText(preset['method'])
            self.min_disparity.setValue(preset['min_disparity'])
            self.num_disparities.setValue(preset['num_disparities'])
            self.block_size.setValue(preset['block_size'])
            
            preset_names = ['纹理丰富场景', '纹理较少场景', '近距离场景', '远距离场景']
            self.log_add(f"[OK] 应用预设参数: {preset_names[preset_id]}")

    def view_point_cloud(self):
        """
        查看点云
        如果当前有生成的点云，直接查看
        否则打开文件选择对话框，选择要查看的PLY文件
        """
        try:
            import open3d as o3d
        except ImportError:
            self.log_add("[ERR] Open3D 库未安装，请运行: pip install open3d")
            QMessageBox.warning(self, "错误", "Open3D 库未安装，请运行: pip install open3d")
            return

        pcd = None

        # 如果当前有生成的点云，直接使用
        if self.point_cloud is not None and self.point_colors is not None:
            self.log_add("[INFO] 查看当前生成的点云")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
            pcd.colors = o3d.utility.Vector3dVector(self.point_colors / 255.0)  # 归一化到0-1
        else:
            # 打开文件选择对话框，选择PLY文件
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择点云文件", PLY_DIR, "PLY文件 (*.ply);;所有文件 (*.*)", options=options
            )

            if not file_path:
                return

            self.log_add(f"[INFO] 加载点云文件: {file_path}")
            try:
                pcd = o3d.io.read_point_cloud(file_path)
                if pcd.is_empty():
                    self.log_add("[ERR] 点云文件为空")
                    QMessageBox.warning(self, "错误", "点云文件为空")
                    return
            except Exception as e:
                self.log_add(f"[ERR] 加载点云文件失败: {str(e)}")
                QMessageBox.warning(self, "错误", f"加载点云文件失败: {str(e)}")
                return

        if pcd is not None:
            # 点云下采样，减少点云数量
            if len(pcd.points) > 50000:
                self.log_add("[INFO] 点云数量过多，进行下采样")
                pcd = pcd.voxel_down_sample(voxel_size=2.0)
                self.log_add(f"[INFO] 下采样后点云数量: {len(pcd.points)}")

            # 统计滤波，去除离群点
            try:
                pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                self.log_add(f"[INFO] 统计滤波后点云数量: {len(pcd.points)}")
            except Exception as e:
                self.log_add(f"[WARN] 统计滤波失败: {str(e)}")

            # 半径滤波，去除稀疏区域的点
            try:
                pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=5.0)
                self.log_add(f"[INFO] 半径滤波后点云数量: {len(pcd.points)}")
            except Exception as e:
                self.log_add(f"[WARN] 半径滤波失败: {str(e)}")

            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="点云查看器", width=800, height=600)
            vis.add_geometry(pcd)

            # 添加坐标系
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])
            vis.add_geometry(coordinate_frame)

            # 设置点大小和背景
            render_option = vis.get_render_option()
            render_option.point_size = 1.0  # 减小点大小，减少重叠
            render_option.background_color = [0.0, 0.0, 0.0]  # 黑色背景

            # 运行可视化
            self.log_add("[OK] 点云查看器已打开")
            vis.run()
            vis.destroy_window()

    def save_point_cloud(self):
        """
        保存点云为PLY格式
        """
        if self.point_cloud is None:
            return

        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        ply_path = os.path.join(PLY_DIR, f"point_cloud_{ts}.ply")

        try:
            with open(ply_path, 'w') as f:
                # PLY 头部
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(self.point_cloud)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")

                # 写入点云数据
                for i in range(len(self.point_cloud)):
                    x, y, z = self.point_cloud[i]
                    r, g, b = self.point_colors[i] if self.point_colors is not None else (255, 255, 255)
                    f.write(f"{x:.3f} {y:.3f} {z:.3f} {int(r)} {int(g)} {int(b)}\n")

            self.log_add(f"[OK] 点云已保存到: {ply_path}")
        except Exception as e:
            self.log_add(f"[ERR] 保存点云失败: {str(e)}")
