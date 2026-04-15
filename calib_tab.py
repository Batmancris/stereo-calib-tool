# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import time
from datetime import datetime  # <-- 鏃堕棿鎴?
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QMessageBox
)

from utils_img import split_sbs, to_pixmap_fit
from config import RECORD_DIR, YAML_DIR
from ui_theme import create_page_header


def make_object_points(cols, rows, square_mm):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_mm)
    return objp


def find_corners(gray, pattern_size):
    """
    妫€娴嬫鐩樻牸瑙掔偣
    
    鍙傛暟锛?
        gray: 鐏板害鍥惧儚
        pattern_size: 妫嬬洏鏍兼ā寮忓昂瀵?(cols, rows)
    
    杩斿洖锛?
        (ok, corners): (鏄惁鎴愬姛, 瑙掔偣鍧愭爣)
    """
    cols, rows = pattern_size

    # 鍥惧儚棰勫鐞嗭細楂樻柉妯＄硦锛屽噺灏戝櫔澹?
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 鏇寸ǔ鐨?SB 绠楁硶锛堝鏋?OpenCV 鏀寔锛?
    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            ok, corners = cv2.findChessboardCornersSB(blurred, (cols, rows))
            if ok:
                # 涓嶉渶瑕侀澶栫殑浜氬儚绱犵簿纭寲锛孲B 绠楁硶宸茬粡寰堢簿纭?
                return True, corners.astype(np.float32)
        except Exception as e:
            # 濡傛灉 SB 绠楁硶澶辫触锛屽洖閫€鍒颁紶缁熺畻娉?
            pass

    # 浼犵粺瑙掔偣妫€娴嬬畻娉?
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH | 
        cv2.CALIB_CB_NORMALIZE_IMAGE |
        cv2.CALIB_CB_FAST_CHECK  # 蹇€熸鏌ワ紝鎻愰珮妫€娴嬮€熷害
    )
    
    ok, corners = cv2.findChessboardCorners(blurred, (cols, rows), flags)
    if not ok:
        return False, None

    # 浜氬儚绱犵簿纭寲锛屼娇鐢ㄦ洿灏忕殑绐楀彛鎻愰珮閫熷害
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
    corners2 = cv2.cornerSubPix(blurred, corners, (7, 7), (-1, -1), term)
    return True, corners2


# ---------------- 鏂板锛氫粠澶ч噺 pairs 閲岄€夆€滆川閲忓ソ + 鍒嗘暎鈥濈殑瀛愰泦 ----------------
def _corners_to_xy(corners) -> np.ndarray:
    # corners: (N,1,2) or (N,2)
    c = np.asarray(corners, dtype=np.float32)
    return c.reshape(-1, 2)


def _homography_rms(grid_xy: np.ndarray, img_xy: np.ndarray) -> float:
    """
    鐢ㄥ崟搴旀€ф妸妫嬬洏鐞嗘兂缃戞牸 -> 鍥惧儚瑙掔偣锛岃绠桼MS娈嬪樊锛堣秺灏忚秺濂斤級銆?
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
    璁＄畻缃戞牸鐩搁偦鐐归棿璺濈殑鈥滃彉寮傜郴鏁扳€?std/mean)锛岃秺灏忚秺濂姐€?
    鐢ㄤ簬妫€娴嬫ā绯?璇瀵艰嚧鐨勭綉鏍间笉鍧囧寑銆?
    """
    pts = img_xy.reshape(rows, cols, 2)  # 娉ㄦ剰锛歠indChessboardCorners杩斿洖椤哄簭閫氬父鏄浼樺厛
    # 琛屾柟鍚戠浉閭昏窛绂?
    dx = np.linalg.norm(pts[:, 1:, :] - pts[:, :-1, :], axis=2).reshape(-1)
    # 鍒楁柟鍚戠浉閭昏窛绂?
    dy = np.linalg.norm(pts[1:, :, :] - pts[:-1, :, :], axis=2).reshape(-1)
    d = np.concatenate([dx, dy], axis=0)
    m = float(np.mean(d)) + 1e-9
    s = float(np.std(d))
    return float(s / m)


def _pair_quality_features(cols: int, rows: int, cL, cR, image_size):
    """
    杈撳嚭锛?
      quality_score: 瓒婂皬瓒婂ソ
      feat: 鐢ㄤ簬鈥滃鏍锋€ч噰鏍封€濈殑鐗瑰緛鍚戦噺锛堝綊涓€鍖栵級
    """
    w, h = float(image_size[0]), float(image_size[1])

    xyL = _corners_to_xy(cL)
    xyR = _corners_to_xy(cR)

    # 鐞嗘兂缃戞牸鍧愭爣锛堝彧鐢ㄤ簬homography鎷熷悎锛屼笉闇€瑕並锛?
    grid = np.stack(np.meshgrid(np.arange(cols, dtype=np.float32),
                                np.arange(rows, dtype=np.float32)), axis=-1).reshape(-1, 2)

    # 1) homography鎷熷悎娈嬪樊锛堣秺灏忚秺濂斤級
    rmsH_L = _homography_rms(grid, xyL)
    rmsH_R = _homography_rms(grid, xyR)

    # 2) 缃戞牸闂磋窛涓€鑷存€э紙瓒婂皬瓒婂ソ锛?
    cvL = _grid_spacing_cv(cols, rows, xyL)
    cvR = _grid_spacing_cv(cols, rows, xyR)

    # 3) 宸﹀彸涓€鑷存€э細瑙嗗樊鍒嗗竷鐨剆td锛堣秺灏忚秺濂斤級
    disp = xyL[:, 0] - xyR[:, 0]
    ydiff = xyL[:, 1] - xyR[:, 1]
    std_disp = float(np.std(disp))
    std_ydiff = float(np.std(ydiff))

    # 4) 澶氭牱鎬х壒寰侊細涓績浣嶇疆銆侀潰绉昂搴︺€佹棆杞€佸钩鍧囪宸?
    minL = np.min(xyL, axis=0); maxL = np.max(xyL, axis=0)
    box_w = float(maxL[0] - minL[0] + 1e-9)
    box_h = float(maxL[1] - minL[1] + 1e-9)
    area = box_w * box_h

    center = np.mean(xyL, axis=0)
    cx = float(center[0] / w)
    cy = float(center[1] / h)
    a = float(np.log(area / (w * h) + 1e-9))  # log灏哄害鏇寸ǔ

    # 鏃嬭浆锛氱敤绗竴琛屾柟鍚戝悜閲忎及涓€涓搴?
    # 鍙?0,0)->(cols-1,0)鐨勫悜閲忥紙鎸夎鐐规帓搴忓亣璁撅級
    p0 = xyL[0]
    p1 = xyL[cols - 1]
    ang = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) / np.pi  # 褰掍竴鍖栧埌[-1,1]

    md = float(np.mean(disp) / w)  # 褰掍竴鍖?

    feat = np.array([cx, cy, a, ang, md], dtype=np.float32)

    # 璐ㄩ噺鍒嗘暟锛堟潈閲嶆槸缁忛獙鍊硷紝鍚庨潰鍙皟锛?
    quality = (
        1.0 * (rmsH_L + rmsH_R) +
        15.0 * (cvL + cvR) +
        0.20 * std_disp +
        0.20 * std_ydiff
    )
    return float(quality), feat


def select_diverse_subset(payload, pattern, target_n=120, top_pool_factor=2.0):
    """
    浠庡ぇ閲忔爣瀹氬浘鍍忓涓€夋嫨璐ㄩ噺濂戒笖鍒嗗竷鍧囧寑鐨勫瓙闆嗭紝鐢ㄤ簬鎻愰珮鏍囧畾绮惧害
    
    鍙傛暟锛?
        payload: 鍖呭惈鏍囧畾鏁版嵁鐨勫瓧鍏革紝蹇呴』鍖呭惈浠ヤ笅閿細
            - objpoints: 鐗╀綋鐐瑰潗鏍囧垪琛?
            - imgL: 宸︾浉鏈鸿鐐瑰潗鏍囧垪琛?
            - imgR: 鍙崇浉鏈鸿鐐瑰潗鏍囧垪琛?
            - image_size: 鍥惧儚灏哄
            - used: 鏈夋晥鏍囧畾瀵规暟閲?
        pattern: 妫嬬洏鏍兼ā寮忓昂瀵?(cols, rows)
        target_n: 鐩爣閫夋嫨鐨勬爣瀹氬鏁伴噺
        top_pool_factor: 璐ㄩ噺鎺掑簭鍚庣殑鍊欓€夋睜澶у皬鍥犲瓙
    
    杩斿洖锛?
        鏂扮殑payload瀛楀吀锛屽彧鍖呭惈閫夋嫨鐨勫瓙闆嗘暟鎹?
    
    瀹炵幇姝ラ锛?
        1. 鎸夎川閲忓垎鏁版帓搴忥紝鍙栧墠 top_pool 涓珮璐ㄩ噺鏍囧畾瀵?
        2. 鍦?top_pool 涓娇鐢?farthest-point sampling 绠楁硶閫夋嫨 target_n 涓垎甯冨潎鍖€鐨勬爣瀹氬
        3. 缁勮骞惰繑鍥炴柊鐨刾ayload
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

    # 璁＄畻姣忎釜鏍囧畾瀵圭殑璐ㄩ噺鍒嗘暟鍜岀壒寰佸悜閲?
    qualities = np.zeros((N,), dtype=np.float64)
    feats = np.zeros((N, 5), dtype=np.float32)

    for i in range(N):
        q, f = _pair_quality_features(cols, rows, imgL[i], imgR[i], image_size)
        qualities[i] = q  # 璐ㄩ噺鍒嗘暟锛岃秺灏忚秺濂?
        feats[i, :] = f    # 鐗瑰緛鍚戦噺锛岀敤浜庡鏍锋€ц瘎浼?

    # 鎸夎川閲忓垎鏁版帓搴忥紝閫夋嫨鍓?top_pool 涓?
    order = np.argsort(qualities)  # 灏?>澶?
    pool_idx = order[:top_pool]

    # 褰掍竴鍖栫壒寰佸悜閲忥紙浣跨敤鍊欓€夋睜鐨勭粺璁￠噺锛?
    F = feats[pool_idx].astype(np.float64)
    mu = F.mean(axis=0)
    sd = F.std(axis=0) + 1e-9  # 閬垮厤闄ら浂
    Fn = (F - mu) / sd

    # 浣跨敤 farthest-point sampling (FPS) 绠楁硶閫夋嫨鍒嗗竷鍧囧寑鐨勬爣瀹氬
    selected_pool_pos = []
    selected_pool_pos.append(0)  # 鍏堥€夋嫨璐ㄩ噺鏈€濂界殑
    dist_to_sel = np.full((top_pool,), np.inf, dtype=np.float64)

    for _ in range(1, target_n):
        last = selected_pool_pos[-1]
        # 璁＄畻姣忎釜鐐瑰埌鏈€杩戝凡閫夌偣鐨勮窛绂?
        d = np.linalg.norm(Fn - Fn[last], axis=1)
        dist_to_sel = np.minimum(dist_to_sel, d)
        # 閫夋嫨璺濈鏈€杩滅殑鐐?
        nxt = int(np.argmax(dist_to_sel))
        selected_pool_pos.append(nxt)

    # 杞崲涓哄師濮嬬储寮?
    sel_idx = pool_idx[np.array(selected_pool_pos, dtype=np.int64)]
    sel_idx = sel_idx.tolist()

    # 缁勮鏂扮殑payload锛屽彧鍖呭惈閫夋嫨鐨勫瓙闆?
    new_payload = dict(payload)
    new_payload["objpoints"] = [objpoints[i] for i in sel_idx]
    new_payload["imgL"] = [imgL[i] for i in sel_idx]
    new_payload["imgR"] = [imgR[i] for i in sel_idx]
    new_payload["used"] = len(sel_idx)
    return new_payload
# ---------------- 鏂板缁撴潫 ----------------


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

    def __init__(self, payload, cols, rows, square_mm, yaml_dir):
        super().__init__()
        self.payload = payload
        self.cols = int(cols)
        self.rows = int(rows)
        self.square_mm = float(square_mm)
        self.yaml_dir = yaml_dir

    def run(self):
        try:
            # 楠岃瘉 payload 鏁版嵁
            if not self.payload:
                raise ValueError("Empty payload received")
                
            objpoints = self.payload.get("objpoints")
            imgL = self.payload.get("imgL")
            imgR = self.payload.get("imgR")
            image_size = self.payload.get("image_size")
            
            if not all([objpoints, imgL, imgR, image_size]):
                raise ValueError("Incomplete payload data")
                
            if len(objpoints) < 10:
                raise ValueError(f"Not enough calibration pairs: {len(objpoints)}")

            def ts():
                return datetime.now().strftime("%H:%M:%S")

            def log(msg):
                self.log_signal.emit(f"[{ts()}] {msg}")

            t_all = datetime.now()

            # ---------- left ----------
            log(f"[INFO] Calibrating left camera... (pairs={len(objpoints)}, image_size={image_size})")
            t0 = datetime.now()
            try:
                rmsL, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgL, image_size, None, None)
                log(f"[LEFT] RMS reproj err: {rmsL:.4f} | time={(datetime.now()-t0).total_seconds():.2f}s")
            except Exception as e:
                log(f"[ERR] Left camera calibration failed: {str(e)}")
                raise

            # ---------- right ----------
            log("[INFO] Calibrating right camera...")
            t0 = datetime.now()
            try:
                rmsR, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgR, image_size, None, None)
                log(f"[RIGHT] RMS reproj err: {rmsR:.4f} | time={(datetime.now()-t0).total_seconds():.2f}s")
            except Exception as e:
                log(f"[ERR] Right camera calibration failed: {str(e)}")
                raise

            # ---------- stereo ----------
            log("[INFO] Stereo calibration (fix intrinsics)...")
            flags = cv2.CALIB_FIX_INTRINSIC
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

            t0 = datetime.now()
            try:
                rmsS, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                    objpoints, imgL, imgR, K1, D1, K2, D2, image_size,
                    criteria=criteria, flags=flags
                )
                baseline = float(np.linalg.norm(T))
                log(f"[STEREO] RMS reproj err: {rmsS:.4f} | time={(datetime.now()-t0).total_seconds():.2f}s")
                log(f"[STEREO] T (mm): {T.ravel()}  | baseline(mm)={baseline:.3f}")
            except Exception as e:
                log(f"[ERR] Stereo calibration failed: {str(e)}")
                raise

            # ---------- rectify ----------
            log("[INFO] stereoRectify...")
            t0 = datetime.now()
            try:
                R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                    K1, D1, K2, D2, image_size, R, T,
                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
                )
                log(f"[RECTIFY] done | time={(datetime.now()-t0).total_seconds():.2f}s")
            except Exception as e:
                log(f"[ERR] Stereo rectification failed: {str(e)}")
                raise

            # ---------- save ----------
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_yaml = os.path.join(
                self.yaml_dir,
                f"stereo_calib_{ts_str}_stRMS{rmsS:.2f}_B{baseline:.1f}mm.yaml"
            )

            log(f"[INFO] Saving YAML -> {out_yaml}")
            t0 = datetime.now()
            try:
                # 纭繚杈撳嚭鐩綍瀛樺湪
                os.makedirs(os.path.dirname(out_yaml), exist_ok=True)
                
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
            except Exception as e:
                log(f"[ERR] Failed to save calibration file: {str(e)}")
                raise

            log(f"[TOTAL] do_calibrate finished | total_time={(datetime.now()-t_all).total_seconds():.2f}s")
            self.done_signal.emit(out_yaml)

        except Exception as e:
            error_msg = f"Calibration failed: {str(e)}"
            log(f"[ERR] {error_msg}")
            self.err_signal.emit(error_msg)


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
        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(14)
        lay.addWidget(create_page_header("双目标定工作台", "从录制视频中筛选高质量棋盘格对，完成双目标定并导出 YAML。这个页面偏工程化，我把层次和留白做得更清楚一些。", accent="#d8a35d"))

        row = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("\u9009\u62e9\u4e00\u4e2a record_*.avi")
        row.addWidget(self.video_edit)

        btn_pick = QPushButton("\u9009\u62e9\u89c6\u9891")
        btn_pick.clicked.connect(self.pick_video)
        row.addWidget(btn_pick)

        lay.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("\u5185\u89d2\u70b9 cols:"))
        self.cols = QSpinBox()
        self.cols.setRange(3, 30)
        self.cols.setValue(11)
        row2.addWidget(self.cols)

        row2.addWidget(QLabel("rows:"))
        self.rows = QSpinBox()
        self.rows.setRange(3, 30)
        self.rows.setValue(8)
        row2.addWidget(self.rows)

        row2.addWidget(QLabel("\u683c\u957f(mm):"))
        self.square = QDoubleSpinBox()
        self.square.setRange(1, 200)
        self.square.setValue(30.0)
        row2.addWidget(self.square)

        row2.addWidget(QLabel("\u6700\u5927\u5bf9\u6570:"))
        self.max_pairs = QSpinBox()
        self.max_pairs.setRange(10, 300)
        self.max_pairs.setValue(80)
        row2.addWidget(self.max_pairs)

        row2.addWidget(QLabel("\u8df3\u5e27:"))
        self.skip = QSpinBox()
        self.skip.setRange(1, 30)
        self.skip.setValue(3)
        row2.addWidget(self.skip)

        self.btn_scan = QPushButton("\u626b\u63cf\u89d2\u70b9")
        self.btn_scan.clicked.connect(self.start_scan)
        row2.addWidget(self.btn_scan)

        self.btn_calib = QPushButton("\u4e00\u952e\u6807\u5b9a\u5e76\u4fdd\u5b58 YAML")
        self.btn_calib.clicked.connect(self.do_calibrate)
        row2.addWidget(self.btn_calib)

        lay.addLayout(row2)

        self.vis_label = QLabel("角点检测可视化（左右并排）")
        self.vis_label.setObjectName("ImagePanel")
        self.vis_label.setMinimumSize(1280, 480)
        self.vis_label.setStyleSheet("border:1px solid gray; background:black;")
        lay.addWidget(self.vis_label)

        self.log = QTextEdit()
        self.log.setObjectName("LogPanel")
        self.log.setReadOnly(True)
        self.log.setFixedHeight(220)
        lay.addWidget(self.log)

        scroll_area.setWidget(container)
        main_lay = QVBoxLayout(self)
        main_lay.addWidget(scroll_area)

    def log_add(self, s):
        self.log.append(s)

    def pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "\u9009\u62e9\u89c6\u9891", RECORD_DIR, "Video (*.avi *.mp4 *.mkv)")
        if p:
            self.video_edit.setText(p)

    def start_scan(self):
        vp = self.video_edit.text().strip()
        if not vp or not os.path.isfile(vp):
            QMessageBox.warning(self, "\u9519\u8bef", "\u8bf7\u5148\u9009\u62e9\u6709\u6548\u89c6\u9891\u6587\u4ef6")
            return

        # 濡傛灉鏍囧畾绾跨▼鍦ㄨ窇锛屽厛涓嶈鎵紙閬垮厤璧勬簮/鐘舵€佹贩涔憋級
        if self.calib_thread is not None and self.calib_thread.isRunning():
            QMessageBox.warning(self, "\u63d0\u793a", "\u6b63\u5728\u6807\u5b9a\u4e2d\uff0c\u8bf7\u7b49\u5f85\u6807\u5b9a\u5b8c\u6210\u540e\u518d\u626b\u63cf\u3002")
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
            self.log_add("[HINT] \u6709\u6548\u5e27\u504f\u5c11\uff1a\u591a\u534a\u662f\u68cb\u76d8\u4e0d\u6e05\u6670\u3001\u59ff\u6001\u4e0d\u591f\u4e30\u5bcc\uff0c\u6216\u89d2\u70b9\u6570\u8bbe\u7f6e\u4e0d\u5bf9\u3002")
            return

        # -------- 鏂板锛氳嚜鍔ㄤ粠澶ч噺pairs閲岄€夆€滆川閲忓ソ+鍒嗘暎鈥濈殑瀛愰泦 --------
        pattern = (int(self.cols.value()), int(self.rows.value()))
        total = int(payload["used"])

        # 浣犲彲浠ユ妸 120 鏀规垚 80/100/150
        target_n = min(80, total)

        if total > target_n:
            self.log_add(f"[INFO] Selecting diverse subset for calibration: {target_n}/{total} ...")
            self.payload = select_diverse_subset(self.payload, pattern, target_n=target_n, top_pool_factor=2.0)
            self.log_add(f"[OK] Subset selected. used={self.payload['used']} (from {total})")
        # -------- 鏂板缁撴潫 --------

    def do_calibrate(self):
        if self.payload is None or self.payload.get("used", 0) < 15:
            QMessageBox.warning(self, "\u9519\u8bef", "\u6709\u6548\u5e27\u592a\u5c11\uff0c\u5efa\u8bae\u81f3\u5c11 15 \u7ec4\u540e\u518d\u5f00\u59cb\u6807\u5b9a\u3002")
            return

        # 濡傛灉鎵弿绾跨▼杩樺湪璺戯紝涓嶅缓璁悓鏃舵爣瀹?
        if self.scan_thread is not None and self.scan_thread.isRunning():
            QMessageBox.warning(self, "\u63d0\u793a", "\u6b63\u5728\u626b\u63cf\u89d2\u70b9\u4e2d\uff0c\u8bf7\u7b49\u626b\u63cf\u7ed3\u675f\u540e\u518d\u5f00\u59cb\u6807\u5b9a\u3002")
            return

        # 闃查噸澶嶇偣鍑?
        if self.calib_thread is not None and self.calib_thread.isRunning():
            QMessageBox.information(self, "\u63d0\u793a", "\u6807\u5b9a\u6b63\u5728\u8fdb\u884c\u4e2d\uff0c\u8bf7\u7b49\u5f85\u5b8c\u6210\u3002")
            return

        # UI锛氱鐢ㄦ寜閽紝閬垮厤閲嶅瑙﹀彂
        self.btn_calib.setEnabled(False)
        self.btn_scan.setEnabled(False)

        self.calib_thread = CalibThread(
            payload=self.payload,
            cols=self.cols.value(),
            rows=self.rows.value(),
            square_mm=self.square.value(),
            yaml_dir=YAML_DIR
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
        QMessageBox.information(self, "\u5b8c\u6210", f"\u6807\u5b9a\u5b8c\u6210\u5e76\u4fdd\u5b58\uff1a\n{out_yaml}")

        self.btn_calib.setEnabled(True)
        self.btn_scan.setEnabled(True)

    def _on_calib_err(self, msg: str):
        self.log_add(f"[ERR] calibrate failed: {msg}")
        QMessageBox.warning(self, "\u9519\u8bef", f"\u6807\u5b9a\u5931\u8d25\uff1a\n{msg}")

        self.btn_calib.setEnabled(True)
        self.btn_scan.setEnabled(True)


