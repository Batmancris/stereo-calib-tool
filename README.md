# Stereo Calibration and High-Precision Tracking Tool

双目标定、矫正、测距与机械臂高精度三维关键点追踪工具。
Stereo calibration, rectification, measurement, and high-precision 3D keypoint tracking for robotic-arm bending experiments.

## Overview | 项目概览
This project is a PyQt5 desktop app built on OpenCV for stereo-camera experiments. It covers live preview, video recording, chessboard-based stereo calibration, rectified playback, interactive measurement, dense disparity exploration, and a sparse high-precision tracking workflow for robotic-arm motion analysis.

这个项目是一个基于 OpenCV 和 PyQt5 的双目视觉桌面工具，覆盖相机预览、视频录制、棋盘格双目标定、双目矫正、交互式测距、稠密视差浏览，以及面向机械臂弯曲实验的稀疏高精度三维关键点追踪流程。

## Highlights | 主要能力
- Stereo preview and recording | 双目预览与录像
- Stereo calibration with YAML export | 双目标定与 YAML 导出
- Rectified playback and manual 3D measurement | 矫正后回放与手工三维测距
- Dense disparity and point cloud inspection | 稠密视差与点云浏览
- High-precision sparse tracking workflow | 高精度稀疏点追踪流程
- CSV export for frame-by-frame coordinates | 逐帧坐标 CSV 导出
- MP4 export for simplified 3D skeleton animation | 三维骨架动画 MP4 导出

## High-Precision Workflow | 高精度实验流程
The new workflow is designed for experiments where only a small set of landmarks matters. Instead of treating dense disparity as the main source of truth, it tracks left/right correspondences for fixed keypoints and triangulates them directly from the rectified stereo projection matrices.

新的实验流程面向“少量固定实验点”的场景，不再把整张视差图作为主依据，而是直接维护左右图关键点对应关系，并使用矫正后的双目投影矩阵进行三角化，从而得到更稳定、可控的高精度三维坐标。

### Tracking pipeline | 追踪步骤
1. Load the stereo YAML file. | 加载双目标定 YAML。
2. Open a recorded experiment video. | 打开实验录像。
3. Freeze a clear frame in the `3D Perception` tab. | 在“三维感知”页冻结一帧清晰图像。
4. Initialize `P01 ~ P15` on the left image. | 在左图初始化 `P01 ~ P15`。
5. Review or correct the suggested right-image correspondences. | 检查并修正右图建议对应点。
6. Save the frame. | 保存当前帧。
7. Track the next frame automatically and correct low-confidence points. | 自动跟踪下一帧，并修正低置信度点。
8. Export CSV and the 3D animation after enough frames are collected. | 在收集足够帧后导出 CSV 和 3D 动画。

## Why Sparse Tracking Instead of Dense Disparity | 为什么主流程不用整张视差图
Dense disparity is still useful for visualization, but robotic-arm bending experiments usually care about a limited number of stable landmarks. Sparse stereo correspondences plus triangulation are often more robust when the scene contains dark backgrounds, highlights, repeated structures, or reflective surfaces.

稠密视差图依然适合做可视化，但机械臂弯曲实验通常只关心少量稳定实验点。在黑色背景、反光表面、重复结构较多的场景里，稀疏对应点加三角化通常比整图视差更稳，也更容易控制误差传播。

## CSV Output | CSV 输出字段
- `frame_idx`
- `timestamp_sec`
- `point_id`
- `u_left`, `v_left`
- `u_right`, `v_right`
- `x_mm`, `y_mm`, `z_mm`
- `track_status`
- `confidence`

## Accuracy Notes | 精度建议
- Calibration quality is the first priority. | 标定质量永远是第一优先级。
- Keep `SPLIT_OFFSET` and `SPLIT_GAP` consistent with the real stereo split. | `SPLIT_OFFSET` 和 `SPLIT_GAP` 必须与真实拼接边界一致。
- Prefer stable corners or textured landmarks over specular highlights. | 尽量选择稳定角点或纹理点，不要选强反光区域。
- Watch the average `|vL-vR|` when saving a frame. | 保存帧时重点关注平均 `|vL-vR|`。
- Always review low-confidence matches manually. | 对低置信度匹配一定要人工复核。

## Dependencies | 依赖
- Python 3.9+
- opencv-python
- numpy
- pyqt5
- matplotlib
- open3d
- ffmpeg

Install example | 安装示例:
```bash
pip install opencv-python numpy pyqt5 matplotlib open3d
```

## Run | 运行
```bash
python main.py
```

## Repository Layout | 目录说明
- `main.py`: application entry point | 主程序入口
- `preview_record_tab.py`: preview and recording UI | 预览与录制页面
- `calib_tab.py`: stereo calibration workflow | 双目标定页面
- `rectify_tab.py`: rectified measurement workflow | 矫正与测距页面
- `perception_3d_tab.py`: high-precision tracking and export workflow | 高精度追踪与导出页面
- `stereo_calibrate_from_video.py`: calibration helper script for recorded stereo videos | 录像标定辅助脚本
- `utils_img.py`: image splitting helpers | 图像切分辅助函数
- `config.py`: path, device, and split configuration | 路径、设备与切分配置

## Status | 当前状态
The branch `update-readme` contains the new tracking workflow, integrated homepage-style README content, and related configuration updates. It is intended to be reviewed through a pull request before merging.

当前 `update-readme` 分支已经包含新的高精度追踪工作流、项目首页风格 README，以及相关配置更新，适合通过 Pull Request 审核后再合并。
