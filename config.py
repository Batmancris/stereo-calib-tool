# -*- coding: utf-8 -*-
import os

# ===== 你需要改的 =====
FFMPEG_EXE = r"C:\Users\12548\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
DEVICE_NAME = "DECXIN Camera"
RECORD_DIR = r"E:\research\0\stero\record"
YAML_DIR = r"E:\research\0\stero\yaml"

SBS_W, SBS_H = 3840, 1080
FPS = 30

PREVIEW_W, PREVIEW_H = 1280, 360
PREVIEW_Q = 7

SPLIT_OFFSET = 0
SPLIT_GAP = 0
# =====================

os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(YAML_DIR, exist_ok=True)
