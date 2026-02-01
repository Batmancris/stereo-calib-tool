# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap

def split_sbs(frame_bgr, offset=0, gap=0):
    h, w = frame_bgr.shape[:2]
    cut = w // 2 + int(offset)
    cut = max(1, min(w - 1, cut))
    gap = max(0, int(gap))
    left_end = max(1, cut - gap // 2)
    right_start = min(w - 1, cut + (gap - gap // 2))
    return frame_bgr[:, :left_end], frame_bgr[:, right_start:]

def to_pixmap_fit(bgr, label: QLabel) -> QPixmap:
    if bgr is None:
        return QPixmap()
    target_w, target_h = max(1, label.width()), max(1, label.height())
    h, w = bgr.shape[:2]
    s = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, nw, nh, 3 * nw, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)
