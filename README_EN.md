# Stereo Calibration and High-Precision Tracking Tool

## Overview
This project is a stereo-vision desktop application built with Python, OpenCV, and PyQt5. It supports stereo preview, recording, calibration, rectification, interactive measurement, and a high-precision 3D keypoint tracking workflow for robotic arm bending experiments.

## Features
- Stereo camera preview and recording
- Chessboard-based stereo calibration with YAML export
- Rectified playback and interactive 3D distance measurement
- Dense disparity and point-cloud visualization
- High-precision experiment workflow for sparse 3D keypoints
  - Manual initialization of 15 keypoints on the first frame
  - Left-view tracking using `calcOpticalFlowPyrLK`
  - Local right-view matching around the epipolar line
  - Manual correction for precision-critical frames
  - Direct triangulation with rectified projection matrices `P1/P2`
  - CSV export for per-frame 3D coordinates
  - MP4 export for a simplified 3D skeleton animation

## Why this workflow avoids dense disparity as the main path
In robotic arm bending experiments, the real goal is accurate 3D trajectories for a small set of landmarks. Sparse stereo correspondences plus triangulation are usually more stable and more precise than relying on a full dense disparity image, especially under reflection, dark backgrounds, or weak texture.

## Dependencies
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

## Run
```bash
python main.py
```

## High-Precision Tracking Workflow
1. Complete stereo calibration and load the YAML file.
2. Open a recorded experiment video in the `3D Perception` tab.
3. Freeze a clear frame.
4. Start first-frame initialization.
5. Click `P01 ~ P15` in the left image.
6. Review and correct the suggested correspondences in the right image.
7. Save the current frame.
8. Track the next frame automatically and review low-confidence points.
9. Export CSV and 3D animation after enough frames are saved.

## CSV Output Columns
- `frame_idx`
- `timestamp_sec`
- `point_id`
- `u_left`, `v_left`
- `u_right`, `v_right`
- `x_mm`, `y_mm`, `z_mm`
- `track_status`
- `confidence`

## Accuracy Notes
- Calibration quality is the first priority.
- Keep `SPLIT_OFFSET` and `SPLIT_GAP` consistent with the actual stereo split.
- Prefer stable corners or textured landmarks over bright specular highlights.
- Check the average `|vL-vR|` reported in the log when saving a frame.
- Manually correct low-confidence matches, especially during occlusion and strong reflection.

## Main Files
- `main.py`: application entry point
- `preview_record_tab.py`: preview and recording UI
- `calib_tab.py`: calibration workflow
- `rectify_tab.py`: rectification and manual measurement
- `perception_3d_tab.py`: high-precision tracking and export workflow
- `utils_img.py`: image split and display helpers
- `config.py`: paths, device, and resolution settings
