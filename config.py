# -*- coding: utf-8 -*-
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== ????? =====
FFMPEG_EXE = r"C:\Users\b\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
DEVICE_NAME = "DECXIN Camera"
RECORD_DIR = os.path.join(BASE_DIR, "record")
YAML_DIR = os.path.join(BASE_DIR, "yaml")
PLY_DIR = os.path.join(BASE_DIR, "ply")

SBS_W, SBS_H = 3840, 1080
FPS = 30

PREVIEW_W, PREVIEW_H = 1280, 360
PREVIEW_Q = 7

SPLIT_OFFSET = 0
SPLIT_GAP = 0
# =====================

os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(YAML_DIR, exist_ok=True)
os.makedirs(PLY_DIR, exist_ok=True)
