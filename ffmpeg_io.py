# -*- coding: utf-8 -*-
import os
import time
import subprocess
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from config import (
    FFMPEG_EXE, DEVICE_NAME,
    SBS_W, SBS_H, FPS,
    PREVIEW_W, PREVIEW_H, PREVIEW_Q
)

class FFmpegPreviewThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)   # BGR
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.proc = None
        self.running = False
        self.recording = False

    def start_stream(self, record=False, record_path=None):
        if self.proc is not None:
            return
        if not os.path.isfile(FFMPEG_EXE):
            self.log_signal.emit(f"[ERR] ffmpeg not found: {FFMPEG_EXE}")
            return

        self.recording = bool(record)

        base_in = [
            FFMPEG_EXE,
            "-hide_banner",
            "-loglevel", "warning",
            "-f", "dshow",
            "-video_size", f"{SBS_W}x{SBS_H}",
            "-framerate", str(FPS),
            "-vcodec", "mjpeg",
            "-i", f"video={DEVICE_NAME}",
        ]

        # 预览：MJPEG image2pipe（不能写 -vframes 0）
        preview_out = [
            "-filter_complex", f"[0:v]scale={PREVIEW_W}:{PREVIEW_H}[vprev]",
            "-map", "[vprev]",
            "-an",
            "-vcodec", "mjpeg",
            "-q:v", str(PREVIEW_Q),
            "-f", "image2pipe",
            "pipe:1",
        ]

        cmd = base_in + preview_out

        # 录制：原始 0:v 直接 copy 到 avi（避免 filtergraph + copy 冲突）
        if self.recording and record_path:
            cmd += [
                "-map", "0:v",
                "-c:v", "copy",
                "-f", "avi",
                record_path,
            ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**7,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        self.running = True
        self.start()

    def stop_stream(self):
        self.running = False
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = None

    def run(self):
        SOI, EOI = b"\xff\xd8", b"\xff\xd9"
        buf = bytearray()

        t0 = time.time()
        cnt = 0

        try:
            while self.running and self.proc and self.proc.stdout:
                try:
                    chunk = self.proc.stdout.read(4096)
                    if not chunk:
                        try:
                            err = self.proc.stderr.read().decode(errors="ignore")
                            if err.strip():
                                self.log_signal.emit("[FFMPEG-ERR]\n" + err)
                        except Exception as e:
                            self.log_signal.emit(f"[ERR] Error reading stderr: {str(e)}")
                        break

                    buf.extend(chunk)

                    while True:
                        s = buf.find(SOI)
                        if s < 0:
                            if len(buf) > 2_000_000:
                                del buf[:-1024]
                            break
                        e = buf.find(EOI, s + 2)
                        if e < 0:
                            if s > 0:
                                del buf[:s]
                            break

                        jpg = bytes(buf[s:e + 2])
                        del buf[:e + 2]

                        try:
                            arr = np.frombuffer(jpg, dtype=np.uint8)
                            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame is None:
                                continue
                            self.frame_signal.emit(frame)

                            cnt += 1
                            t1 = time.time()
                            if t1 - t0 >= 1.0:
                                self.log_signal.emit(f"[PREVIEW FPS] {cnt/(t1-t0):.1f}")
                                t0, cnt = t1, 0
                        except Exception as e:
                            self.log_signal.emit(f"[ERR] Error processing frame: {str(e)}")
                            continue
                except Exception as e:
                    self.log_signal.emit(f"[ERR] Error reading stream: {str(e)}")
                    break
        except Exception as e:
            self.log_signal.emit(f"[ERR] Unexpected error in FFmpeg thread: {str(e)}")
        finally:
            if self.proc is not None:
                try:
                    self.proc.kill()
                except Exception as e:
                    self.log_signal.emit(f"[ERR] Error killing process: {str(e)}")
            self.proc = None
            self.running = False
            self.log_signal.emit("[INFO] FFmpeg thread stopped")
