# -*- coding: utf-8 -*-
"""
公共工具模块
包含在多个模块中重复使用的函数
"""
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from typing import Optional


def read_mat(fs: cv2.FileStorage, name: str) -> np.ndarray:
    """
    从OpenCV FileStorage中读取矩阵
    
    参数:
        fs: OpenCV FileStorage对象
        name: 矩阵名称
    
    返回:
        读取的矩阵
    
    异常:
        RuntimeError: 如果矩阵不存在
    """
    node = fs.getNode(name)
    if node.empty():
        raise RuntimeError(f"YAML missing: {name}")
    return node.mat()


def bgr_to_qpixmap(bgr: Optional[np.ndarray]) -> QPixmap:
    """
    将BGR图像转换为QPixmap
    
    参数:
        bgr: BGR格式的图像
    
    返回:
        转换后的QPixmap
    """
    if bgr is None:
        return QPixmap()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)
