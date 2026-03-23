# Stereo Calibration and High-Precision Tracking Tool

## 中文说明

### 项目概览
这是一个基于 Python、OpenCV 和 PyQt5 的双目视觉实验工具，用于双目标定、双目矫正、视频录制、交互式测距，以及面向机械臂弯曲实验的高精度关键点三维追踪。

### 当前主要功能
- 双目相机预览与录像
- 棋盘格双目标定与 YAML 导出
- 矫正后双目视频回放与手工测距
- 稠密视差与点云浏览
- 高精度关键点追踪实验模式
  - 首帧手工初始化 15 个点
  - 后续帧使用 `calcOpticalFlowPyrLK` 跟踪左图关键点
  - 右图在极线附近做局部模板匹配，并允许人工修正
  - 使用矫正后的投影矩阵 `P1/P2` 直接三角化，获得每个点的 `X/Y/Z`
  - 导出逐帧 CSV 数据
  - 导出三维骨架动画 MP4

### 为什么高精度模式不依赖整张视差图
机械臂实验更关心少量固定实验点的空间轨迹，而不是整场景的稠密深度。对这类任务，直接维护左右对应关键点并做三角化，通常比依赖整张视差图更稳，也更容易控制误差。

### 主要依赖
- Python 3.9+
- opencv-python
- numpy
- pyqt5
- matplotlib
- open3d
- ffmpeg

安装示例：
```bash
pip install opencv-python numpy pyqt5 matplotlib open3d
```

### 运行方式
```bash
python main.py
```

### 高精度关键点追踪使用流程
1. 先完成双目标定，并加载标定得到的 YAML。
2. 打开实验视频，在“三维感知”页冻结一帧清晰的机械臂图像。
3. 点击“开始首帧初始化”。
4. 在左图依次点击 `P01 ~ P15`，系统会自动给出右图建议点。
5. 在右图逐点修正对应点，保证 `|vL - vR|` 尽量小。
6. 点击“保存当前帧”。
7. 点击“自动跟踪下一帧”，检查低置信度点并修正。
8. 重复保存所需帧后，导出 CSV 和 3D 动画。

### CSV 输出字段
- `frame_idx`
- `timestamp_sec`
- `point_id`
- `u_left`, `v_left`
- `u_right`, `v_right`
- `x_mm`, `y_mm`, `z_mm`
- `track_status`
- `confidence`

### 精度建议
- 优先保证标定质量，尤其是双目外参与重投影误差。
- 使用 `SPLIT_OFFSET` 和 `SPLIT_GAP` 精确切分左右图。
- 选点时尽量点击稳定的结构角点或纹理点，避免纯高光区域。
- 重点关注保存帧时日志中的平均 `|vL-vR|`，这个值越小越好。
- 对自动匹配结果保持人工复核，尤其是遮挡、反光和快速弯曲阶段。

### 项目结构
- `main.py`: 主界面入口
- `preview_record_tab.py`: 预览与录制
- `calib_tab.py`: 标定流程
- `rectify_tab.py`: 矫正与交互测距
- `perception_3d_tab.py`: 三维感知与高精度关键点追踪
- `utils_img.py`: 图像切分与显示辅助函数
- `config.py`: 路径、分辨率与设备配置

## English

### Overview
This project is a stereo-vision desktop tool built with Python, OpenCV, and PyQt5. It supports stereo preview, recording, calibration, rectification, interactive measurement, dense disparity exploration, and a new high-precision 3D keypoint tracking workflow for robotic arm bending experiments.

### Key Features
- Stereo camera preview and recording
- Chessboard-based stereo calibration with YAML export
- Rectified video playback and interactive distance measurement
- Dense disparity and point-cloud inspection
- High-precision keypoint tracking mode
  - Manual initialization of 15 keypoints on the first frame
  - Left-image tracking with `calcOpticalFlowPyrLK`
  - Local right-image matching along the epipolar line, with manual correction
  - Direct triangulation from rectified projection matrices `P1/P2`
  - CSV export for frame-by-frame 3D data
  - MP4 export for a simplified 3D skeleton animation

### Why the high-precision workflow does not rely on dense disparity
For bending experiments, the real target is the 3D trajectory of a small set of fixed landmarks rather than a dense depth map of the entire scene. Tracking stereo correspondences for those landmarks and triangulating them directly is usually more stable and more accurate than relying on a full disparity image.

### Dependencies
- Python 3.9+
- opencv-python
- numpy
- pyqt5
- matplotlib
- open3d
- ffmpeg

Install example:
```bash
pip install opencv-python numpy pyqt5 matplotlib open3d
```

### Run
```bash
python main.py
```

### High-Precision Tracking Workflow
1. Finish stereo calibration and load the generated YAML file.
2. Open an experiment video and freeze a clear frame in the `3D Perception` tab.
3. Click `Start First-Frame Initialization`.
4. Click `P01 ~ P15` in the left image.
5. Review and correct the suggested right-image correspondences.
6. Save the current frame.
7. Track the next frame automatically, then review low-confidence points.
8. Export the CSV data and the 3D animation after enough frames are saved.

### CSV Fields
- `frame_idx`
- `timestamp_sec`
- `point_id`
- `u_left`, `v_left`
- `u_right`, `v_right`
- `x_mm`, `y_mm`, `z_mm`
- `track_status`
- `confidence`
