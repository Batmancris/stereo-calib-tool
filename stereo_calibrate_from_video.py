# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np

# ===================== 你需要改的参数 =====================
VIDEO_PATH = r"E:\research\0\stero\record_20260131_152954.avi"
OUT_YAML   = r"E:\research\0\stero\stereo_calib.yaml"

# SBS 分割参数（一般不用改）
SPLIT_OFFSET = 0   # 正数切点右移
SPLIT_GAP    = 0   # 中间黑缝像素

# 棋盘内角点数（非常关键！）
# 注意：这里是“内角点”，不是方格数
# 你这块 Q12-400-30 很可能是 12x9 或 11x8（你如果检测失败就改这里）
PATTERN_SIZE = (11, 8)   # (列, 行) 例如 12x9

# 棋盘格边长（单位：毫米）
SQUARE_SIZE_MM = 30.0

# 抽帧策略
MAX_FRAMES_TO_USE = 60   # 最多用多少组有效帧
SKIP = 3                 # 每隔多少帧尝试一次（减少重复姿态）
# =========================================================


def split_sbs(frame_bgr, offset=0, gap=0):
    h, w = frame_bgr.shape[:2]
    cut = w // 2 + int(offset)
    cut = max(1, min(w - 1, cut))
    gap = max(0, int(gap))
    left_end = max(1, cut - gap // 2)
    right_start = min(w - 1, cut + (gap - gap // 2))
    left = frame_bgr[:, :left_end]
    right = frame_bgr[:, right_start:]
    return left, right


def make_object_points(pattern_size, square_size):
    # (0,0,0), (1,0,0), ... 乘以 square_size
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def find_corners(gray, pattern_size):
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return False, None

    # 亚像素
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return True, corners2


def calibrate_single(objpoints, imgpoints, image_size):
    # 常规单目标定
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    return ret, K, D


def main():
    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(VIDEO_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频")

    objp = make_object_points(PATTERN_SIZE, SQUARE_SIZE_MM)

    objpoints = []      # 3D点（同一套给左右）
    imgpoints_l = []    # 左图 2D
    imgpoints_r = []    # 右图 2D

    used = 0
    idx = 0
    image_size = None

    print("[INFO] Start scanning video for chessboard corners...")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        idx += 1
        if idx % SKIP != 0:
            continue

        left, right = split_sbs(frame, SPLIT_OFFSET, SPLIT_GAP)
        if image_size is None:
            image_size = (left.shape[1], left.shape[0])  # (w,h)

        gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        ok_l, corners_l = find_corners(gl, PATTERN_SIZE)
        ok_r, corners_r = find_corners(gr, PATTERN_SIZE)

        if ok_l and ok_r:
            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)
            used += 1
            print(f"[OK] pair {used} at frame {idx}")

            # 可视化（按需打开）
            vis = np.hstack([left.copy(), right.copy()])
            cv2.drawChessboardCorners(vis[:, :left.shape[1]], PATTERN_SIZE, corners_l, True)
            cv2.drawChessboardCorners(vis[:, left.shape[1]:], PATTERN_SIZE, corners_r, True)
            cv2.imshow("corners", cv2.resize(vis, (1280, 360)))
            cv2.waitKey(1)

            if used >= MAX_FRAMES_TO_USE:
                break

    cap.release()
    cv2.destroyAllWindows()

    if used < 10:
        raise RuntimeError(f"有效帧太少：{used}（请检查 PATTERN_SIZE 是否正确，或视频里棋盘是否清晰）")

    print(f"[INFO] Collected {used} valid stereo pairs.")
    print("[INFO] Calibrating left camera...")
    rms_l, K1, D1 = calibrate_single(objpoints, imgpoints_l, image_size)
    print(f"[LEFT] RMS reproj err: {rms_l:.4f}")

    print("[INFO] Calibrating right camera...")
    rms_r, K2, D2 = calibrate_single(objpoints, imgpoints_r, image_size)
    print(f"[RIGHT] RMS reproj err: {rms_r:.4f}")

    # 双目标定：固定内参（更稳），只求 R,T,E,F
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    print("[INFO] Stereo calibration (fix intrinsics)...")
    rms_stereo, K1o, D1o, K2o, D2o, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K1, D1, K2, D2, image_size,
        criteria=criteria, flags=flags
    )
    print(f"[STEREO] RMS reproj err: {rms_stereo:.4f}")
    print(f"[STEREO] T (mm): {T.ravel()}  | baseline(mm)={np.linalg.norm(T):.3f}")

    # 立体校正，得到 Q
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    # 保存为 OpenCV FileStorage YAML（最省事、兼容性最好）
    fs = cv2.FileStorage(OUT_YAML, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("pattern_cols", PATTERN_SIZE[0])
    fs.write("pattern_rows", PATTERN_SIZE[1])
    fs.write("square_size_mm", float(SQUARE_SIZE_MM))

    fs.write("K1", K1); fs.write("D1", D1)
    fs.write("K2", K2); fs.write("D2", D2)
    fs.write("R", R);   fs.write("T", T)
    fs.write("E", E);   fs.write("F", F)
    fs.write("R1", R1); fs.write("R2", R2)
    fs.write("P1", P1); fs.write("P2", P2)
    fs.write("Q", Q)
    fs.release()

    print(f"[DONE] Saved calibration to: {OUT_YAML}")

    # 额外：给你一段“如何用标定结果做矫正map”的模板
    print("\n[HINT] Rectification maps example:")
    print("  map1x,map1y = cv2.initUndistortRectifyMap(K1,D1,R1,P1,(w,h),cv2.CV_16SC2)")
    print("  map2x,map2y = cv2.initUndistortRectifyMap(K2,D2,R2,P2,(w,h),cv2.CV_16SC2)")


if __name__ == "__main__":
    main()
