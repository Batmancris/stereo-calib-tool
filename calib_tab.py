# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import time
from datetime import datetime  # <-- 时间戳
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QMessageBox
)

from utils_img import split_sbs, to_pixmap_fit
from config import RECORD_DIR


def make_object_points(cols, rows, square_mm):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_mm)
    return objp


def find_corners(gray, pattern_size):
    cols, rows = pattern_size

    # 更稳的 SB（如果 OpenCV 支持）
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows))
        if ok:
            return True, corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return False, None

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return True, corners2


# ---------------- 新增：从大量 pairs 里选“质量好 + 分散”的子集 ----------------
def _corners_to_xy(corners) -> np.ndarray:
    # corners: (N,1,2) or (N,2)
    c = np.asarray(corners, dtype=np.float32)
    return c.reshape(-1, 2)


def _homography_rms(grid_xy: np.ndarray, img_xy: np.ndarray) -> float:
    """
    用单应性把棋盘理想网格 -> 图像角点，计算RMS残差（越小越好）。
    """
    if grid_xy.shape[0] < 4:
        return float("inf")
    H, _ = cv2.findHomography(grid_xy, img_xy, 0)
    if H is None:
        return float("inf")
    proj = cv2.perspectiveTransform(grid_xy.reshape(1, -1, 2), H).reshape(-1, 2)
    err = proj - img_xy
    rms = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
    return rms


def _grid_spacing_cv(cols: int, rows: int, img_xy: np.ndarray) -> float:
    """
    计算网格相邻点间距的“变异系数”(std/mean)，越小越好。
    用于检测模糊/误检导致的网格不均匀。
    """
    pts = img_xy.reshape(rows, cols, 2)  # 注意：findChessboardCorners返回顺序通常是行优先
    # 行方向相邻距离
    dx = np.linalg.norm(pts[:, 1:, :] - pts[:, :-1, :], axis=2).reshape(-1)
    # 列方向相邻距离
    dy = np.linalg.norm(pts[1:, :, :] - pts[:-1, :, :], axis=2).reshape(-1)
    d = np.concatenate([dx, dy], axis=0)
    m = float(np.mean(d)) + 1e-9
    s = float(np.std(d))
    return float(s / m)


def _pair_quality_features(cols: int, rows: int, cL, cR, image_size):
    """
    输出：
      quality_score: 越小越好
      feat: 用于“多样性采样”的特征向量（归一化）
    """
    w, h = float(image_size[0]), float(image_size[1])

    xyL = _corners_to_xy(cL)
    xyR = _corners_to_xy(cR)

    # 理想网格坐标（只用于homography拟合，不需要K）
    grid = np.stack(np.meshgrid(np.arange(cols, dtype=np.float32),
                                np.arange(rows, dtype=np.float32)), axis=-1).reshape(-1, 2)

    # 1) homography拟合残差（越小越好）
    rmsH_L = _homography_rms(grid, xyL)
    rmsH_R = _homography_rms(grid, xyR)

    # 2) 网格间距一致性（越小越好）
    cvL = _grid_spacing_cv(cols, rows, xyL)
    cvR = _grid_spacing_cv(cols, rows, xyR)

    # 3) 左右一致性：视差分布的std（越小越好）
    disp = xyL[:, 0] - xyR[:, 0]
    ydiff = xyL[:, 1] - xyR[:, 1]
    std_disp = float(np.std(disp))
    std_ydiff = float(np.std(ydiff))

    # 4) 多样性特征：中心位置、面积尺度、旋转、平均视差
    minL = np.min(xyL, axis=0); maxL = np.max(xyL, axis=0)
    box_w = float(maxL[0] - minL[0] + 1e-9)
    box_h = float(maxL[1] - minL[1] + 1e-9)
    area = box_w * box_h

    center = np.mean(xyL, axis=0)
    cx = float(center[0] / w)
    cy = float(center[1] / h)
    a = float(np.log(area / (w * h) + 1e-9))  # log尺度更稳

    # 旋转：用第一行方向向量估一个角度
    # 取(0,0)->(cols-1,0)的向量（按角点排序假设）
    p0 = xyL[0]
    p1 = xyL[cols - 1]
    ang = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) / np.pi  # 归一化到[-1,1]

    md = float(np.mean(disp) / w)  # 归一化

    feat = np.array([cx, cy, a, ang, md], dtype=np.float32)

    # 质量分数（权重是经验值，后面可调）
    quality = (
        1.0 * (rmsH_L + rmsH_R) +
        15.0 * (cvL + cvR) +
        0.20 * std_disp +
        0.20 * std_ydiff
    )
    return float(quality), feat


def select_diverse_subset(payload, pattern, target_n=120, top_pool_factor=2.0):
    """
    从 payload 里选子集：
      1) 按质量分数排序，取前 top_pool
      2) 在 top_pool 里做 farthest-point sampling，选 target_n 个（覆盖多样姿态）
    """
    used = int(payload.get("used", 0))
    if used <= 0:
        return payload

    cols, rows = int(pattern[0]), int(pattern[1])
    image_size = payload["image_size"]

    objpoints = payload["objpoints"]
    imgL = payload["imgL"]
    imgR = payload["imgR"]

    N = len(imgL)
    target_n = int(min(max(10, target_n), N))
    top_pool = int(min(N, max(target_n, int(round(target_n * float(top_pool_factor))))))

    # 计算每个pair的质量+特征
    qualities = np.zeros((N,), dtype=np.float64)
    feats = np.zeros((N, 5), dtype=np.float32)

    for i in range(N):
        q, f = _pair_quality_features(cols, rows, imgL[i], imgR[i], image_size)
        qualities[i] = q
        feats[i, :] = f

    order = np.argsort(qualities)  # 小->大
    pool_idx = order[:top_pool]

    # 归一化特征（用pool统计量）
    F = feats[pool_idx].astype(np.float64)
    mu = F.mean(axis=0)
    sd = F.std(axis=0) + 1e-9
    Fn = (F - mu) / sd

    # FPS：先选质量最好的，然后不断选“离已选集合最远”的点
    selected_pool_pos = []
    selected_pool_pos.append(0)  # pool里第0个就是全局最优quality
    dist_to_sel = np.full((top_pool,), np.inf, dtype=np.float64)

    for _ in range(1, target_n):
        last = selected_pool_pos[-1]
        d = np.linalg.norm(Fn - Fn[last], axis=1)
        dist_to_sel = np.minimum(dist_to_sel, d)
        nxt = int(np.argmax(dist_to_sel))
        selected_pool_pos.append(nxt)

    sel_idx = pool_idx[np.array(selected_pool_pos, dtype=np.int64)]
    sel_idx = sel_idx.tolist()

    # 组装新的payload（只替换这三项）
    new_payload = dict(payload)
    new_payload["objpoints"] = [objpoints[i] for i in sel_idx]
    new_payload["imgL"] = [imgL[i] for i in sel_idx]
    new_payload["imgR"] = [imgR[i] for i in sel_idx]
    new_payload["used"] = len(sel_idx)
    return new_payload
# ---------------- 新增结束 ----------------


class ScanThread(QThread):
    vis_signal = pyqtSignal(np.ndarray, bool, int, int)  # vis_bgr, ok_pair, used, frame_idx
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(object)

    def __init__(self, video_path, pattern, square_mm, max_pairs=80, skip=3):
        super().__init__()
        self.video_path = video_path
        self.pattern = pattern
        self.square_mm = square_mm
        self.max_pairs = max_pairs
        self.skip = skip

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log_signal.emit("[ERR] cannot open video")
            self.done_signal.emit(None)
            return

        objp = make_object_points(self.pattern[0], self.pattern[1], self.square_mm)

        objpoints, imgL, imgR = [], [], []
        used = 0
        idx = 0
        image_size = None

        self.log_signal.emit("[INFO] Start scanning video for chessboard corners...")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            idx += 1
            if idx % self.skip != 0:
                continue

            left, right = split_sbs(frame, 0, 0)
            if image_size is None:
                image_size = (left.shape[1], left.shape[0])

            gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            okL, cL = find_corners(gl, self.pattern)
            okR, cR = find_corners(gr, self.pattern)
            ok_pair = bool(okL and okR)

            vis = np.hstack([left.copy(), right.copy()])
            if okL:
                cv2.drawChessboardCorners(vis[:, :left.shape[1]], self.pattern, cL, True)
            if okR:
                cv2.drawChessboardCorners(vis[:, left.shape[1]:], self.pattern, cR, True)

            if ok_pair:
                objpoints.append(objp)
                imgL.append(cL)
                imgR.append(cR)
                used += 1
                self.log_signal.emit(f"[OK] pair {used} at frame {idx}")

            self.vis_signal.emit(vis, ok_pair, used, idx)

            if used >= self.max_pairs:
                break

        cap.release()
        self.log_signal.emit(f"[INFO] Collected {used} valid stereo pairs.")
        self.done_signal.emit({
            "image_size": image_size,
            "objpoints": objpoints,
            "imgL": imgL,
            "imgR": imgR,
            "used": used
        })


class CalibThread(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str)     # out_yaml
    err_signal = pyqtSignal(str)

    def __init__(self, payload, cols, rows, square_mm, record_dir):
        super().__init__()
        self.payload = payload
        self.cols = int(cols)
        self.rows = int(rows)
        self.square_mm = float(square_mm)
        self.record_dir = record_dir

    def run(self):
        try:
            objpoints = self.payload["objpoints"]
            imgL = self.payload["imgL"]
            imgR = self.payload["imgR"]
            image_size = self.payload["image_size"]

            def ts():
                return datetime.now().strftime("%H:%M:%S")

            def log(msg):
                self.log_signal.emit(f"[{ts()}] {msg}")

            t_all = datetime.now()

            # ---------- left ----------
            log(f"[INFO] Calibrating left camera... (pairs={len(objpoints)}, image_size={image_size})")
            t0 = datetime.now()
            rmsL, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgL, image_size, None, None)
            log(f"[LEFT] RMS reproj err: {rmsL:.4f} | time={(datetime.now()-t0).total_seconds():.2f}s")

            # ---------- right ----------
            log("[INFO] Calibrating right camera...")
            t0 = datetime.now()
            rmsR, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgR, image_size, None, None)
            log(f"[RIGHT] RMS reproj err: {rmsR:.4f} | time={(datetime.now()-t0).total_seconds():.2f}s")

            # ---------- stereo ----------
            log("[INFO] Stereo calibration (fix intrinsics)...")
            flags = cv2.CALIB_FIX_INTRINSIC
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

            t0 = datetime.now()
            rmsS, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgL, imgR, K1, D1, K2, D2, image_size,
                criteria=criteria, flags=flags
            )
            baseline = float(np.linalg.norm(T))
            log(f"[STEREO] RMS reproj err: {rmsS:.4f} | time={(datetime.now()-t0).total_seconds():.2f}s")
            log(f"[STEREO] T (mm): {T.ravel()}  | baseline(mm)={baseline:.3f}")

            # ---------- rectify ----------
            log("[INFO] stereoRectify...")
            t0 = datetime.now()
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                K1, D1, K2, D2, image_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
            )
            log(f"[RECTIFY] done | time={(datetime.now()-t0).total_seconds():.2f}s")

            # ---------- save ----------
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_yaml = os.path.join(
                self.record_dir,
                f"stereo_calib_{ts_str}_stRMS{rmsS:.2f}_B{baseline:.1f}mm.yaml"
            )

            log(f"[INFO] Saving YAML -> {out_yaml}")
            t0 = datetime.now()
            fs = cv2.FileStorage(out_yaml, cv2.FILE_STORAGE_WRITE)
            fs.write("image_width", image_size[0])
            fs.write("image_height", image_size[1])
            fs.write("pattern_cols", self.cols)
            fs.write("pattern_rows", self.rows)
            fs.write("square_size_mm", self.square_mm)
            fs.write("K1", K1); fs.write("D1", D1)
            fs.write("K2", K2); fs.write("D2", D2)
            fs.write("R", R);   fs.write("T", T)
            fs.write("E", E);   fs.write("F", F)
            fs.write("R1", R1); fs.write("R2", R2)
            fs.write("P1", P1); fs.write("P2", P2)
            fs.write("Q", Q)
            fs.release()
            log(f"[DONE] Saved calibration YAML | time={(datetime.now()-t0).total_seconds():.2f}s")

            log(f"[TOTAL] do_calibrate finished | total_time={(datetime.now()-t_all).total_seconds():.2f}s")
            self.done_signal.emit(out_yaml)

        except Exception as e:
            self.err_signal.emit(str(e))


class CalibTab(QWidget):
    def __init__(self):
        super().__init__()
        self.scan_thread = None
        self.calib_thread = None

        self.payload = None
        self.latest_vis = None
        self.latest_ok = False
        self.latest_used = 0
        self.latest_idx = 0

        self._init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh_vis)
        self.timer.start(30)

    def _init_ui(self):
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("选择一个 record_*.avi")
        row.addWidget(self.video_edit)

        btn_pick = QPushButton("选择视频")
        btn_pick.clicked.connect(self.pick_video)
        row.addWidget(btn_pick)

        lay.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("内角点 cols:"))
        self.cols = QSpinBox()
        self.cols.setRange(3, 30)
        self.cols.setValue(11)
        row2.addWidget(self.cols)

        row2.addWidget(QLabel("rows:"))
        self.rows = QSpinBox()
        self.rows.setRange(3, 30)
        self.rows.setValue(8)
        row2.addWidget(self.rows)

        row2.addWidget(QLabel("格长(mm):"))
        self.square = QDoubleSpinBox()
        self.square.setRange(1, 200)
        self.square.setValue(30.0)
        row2.addWidget(self.square)

        row2.addWidget(QLabel("最大对数:"))
        self.max_pairs = QSpinBox()
        self.max_pairs.setRange(10, 300)
        self.max_pairs.setValue(80)
        row2.addWidget(self.max_pairs)

        row2.addWidget(QLabel("跳帧:"))
        self.skip = QSpinBox()
        self.skip.setRange(1, 30)
        self.skip.setValue(3)
        row2.addWidget(self.skip)

        self.btn_scan = QPushButton("扫描角点")
        self.btn_scan.clicked.connect(self.start_scan)
        row2.addWidget(self.btn_scan)

        self.btn_calib = QPushButton("一键标定+保存YAML")
        self.btn_calib.clicked.connect(self.do_calibrate)
        row2.addWidget(self.btn_calib)

        lay.addLayout(row2)

        self.vis_label = QLabel("角点检测可视化（左右并排）")
        self.vis_label.setMinimumSize(1280, 480)
        self.vis_label.setStyleSheet("border:1px solid gray; background:black;")
        lay.addWidget(self.vis_label)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(220)
        lay.addWidget(self.log)

    def log_add(self, s):
        self.log.append(s)

    def pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择视频", RECORD_DIR, "Video (*.avi *.mp4 *.mkv)")
        if p:
            self.video_edit.setText(p)

    def start_scan(self):
        vp = self.video_edit.text().strip()
        if not vp or not os.path.isfile(vp):
            QMessageBox.warning(self, "错误", "请先选择有效视频文件")
            return

        # 如果标定线程在跑，先不让扫（避免资源/状态混乱）
        if self.calib_thread is not None and self.calib_thread.isRunning():
            QMessageBox.warning(self, "提示", "正在标定中，请等待标定完成后再扫描。")
            return

        pattern = (int(self.cols.value()), int(self.rows.value()))
        square_mm = float(self.square.value())

        self.payload = None
        self.scan_thread = ScanThread(
            vp, pattern, square_mm,
            max_pairs=int(self.max_pairs.value()),
            skip=int(self.skip.value())
        )
        self.scan_thread.vis_signal.connect(self.on_vis)
        self.scan_thread.log_signal.connect(self.log_add)
        self.scan_thread.done_signal.connect(self.on_done)
        self.log_add(f"[INFO] pattern={pattern}, square={square_mm}mm")
        self.scan_thread.start()

    def on_vis(self, vis_bgr, ok_pair, used, frame_idx):
        self.latest_vis = vis_bgr
        self.latest_ok = ok_pair
        self.latest_used = used
        self.latest_idx = frame_idx

    def _refresh_vis(self):
        if self.latest_vis is None:
            return
        vis = self.latest_vis.copy()
        txt = f"{'OK' if self.latest_ok else 'NO'}  used={self.latest_used}  frame={self.latest_idx}"
        cv2.putText(vis, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 0) if self.latest_ok else (0, 0, 255), 3)
        self.vis_label.setPixmap(to_pixmap_fit(vis, self.vis_label))

    def on_done(self, payload):
        self.payload = payload
        if payload is None:
            self.log_add("[ERR] scan failed")
            return
        self.log_add(f"[DONE] collected pairs = {payload['used']}")
        if payload["used"] < 15:
            self.log_add("[HINT] 有效帧偏少：多半是棋盘不清晰/姿态不足/角点数写错。")
            return

        # -------- 新增：自动从大量pairs里选“质量好+分散”的子集 --------
        pattern = (int(self.cols.value()), int(self.rows.value()))
        total = int(payload["used"])

        # 你可以把 120 改成 80/100/150
        target_n = min(80, total)

        if total > target_n:
            self.log_add(f"[INFO] Selecting diverse subset for calibration: {target_n}/{total} ...")
            self.payload = select_diverse_subset(self.payload, pattern, target_n=target_n, top_pool_factor=2.0)
            self.log_add(f"[OK] Subset selected. used={self.payload['used']} (from {total})")
        # -------- 新增结束 --------

    def do_calibrate(self):
        if self.payload is None or self.payload.get("used", 0) < 15:
            QMessageBox.warning(self, "错误", "有效帧太少（建议>=15），先扫描收集角点")
            return

        # 如果扫描线程还在跑，不建议同时标定
        if self.scan_thread is not None and self.scan_thread.isRunning():
            QMessageBox.warning(self, "提示", "正在扫描角点中，请等扫描结束再开始标定。")
            return

        # 防重复点击
        if self.calib_thread is not None and self.calib_thread.isRunning():
            QMessageBox.information(self, "提示", "标定正在进行中，请等待完成。")
            return

        # UI：禁用按钮，避免重复触发
        self.btn_calib.setEnabled(False)
        self.btn_scan.setEnabled(False)

        self.calib_thread = CalibThread(
            payload=self.payload,
            cols=self.cols.value(),
            rows=self.rows.value(),
            square_mm=self.square.value(),
            record_dir=RECORD_DIR
        )
        self.calib_thread.log_signal.connect(self.log_add)
        self.calib_thread.done_signal.connect(self._on_calib_done)
        self.calib_thread.err_signal.connect(self._on_calib_err)

        self.log_add("[INFO] Start calibration in background thread...")
        self.calib_thread.start()

    def _on_calib_done(self, out_yaml: str):
        self.log_add(f"[DONE] Saved calibration to: {out_yaml}")
        self.log_add("\n[HINT] Rectification maps example:\n"
                     "  map1x,map1y = cv2.initUndistortRectifyMap(K1,D1,R1,P1,(w,h),cv2.CV_16SC2)\n"
                     "  map2x,map2y = cv2.initUndistortRectifyMap(K2,D2,R2,P2,(w,h),cv2.CV_16SC2)")
        QMessageBox.information(self, "完成", f"标定完成并保存：\n{out_yaml}")

        self.btn_calib.setEnabled(True)
        self.btn_scan.setEnabled(True)

    def _on_calib_err(self, msg: str):
        self.log_add(f"[ERR] calibrate failed: {msg}")
        QMessageBox.warning(self, "错误", f"标定失败：\n{msg}")

        self.btn_calib.setEnabled(True)
        self.btn_scan.setEnabled(True)
