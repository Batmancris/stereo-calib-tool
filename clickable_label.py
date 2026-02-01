# -*- coding: utf-8 -*-
import numpy as np
import cv2

from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QPainter, QPixmap, QImage
from PyQt5.QtWidgets import QWidget


class ClickableImageLabel(QWidget):
    """
    替代旧版 QLabel：
    - 显示一张 pixmap（或BGR图像）
    - 鼠标滚轮缩放（以鼠标位置为中心）
    - 左键拖拽平移
    - 左键点击发射 clicked(u, v)：u/v 是“原图像素坐标”（float->int）
    """

    clicked = pyqtSignal(int, int)

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._hint_text = text

        self._pix = None
        self._img_w = 0
        self._img_h = 0

        self._scale = 1.0
        self._min_scale = 0.05
        self._max_scale = 40.0
        self._offset = QPointF(0.0, 0.0)  # in widget coords

        self._dragging = False
        self._press_pos = QPointF(0.0, 0.0)
        self._last_mouse = QPointF(0.0, 0.0)
        self._moved = False
        self._move_thresh = 4.0  # px threshold to treat as drag

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)

    # ---------- public API ----------
    def set_image_pixmap(self, pixmap: QPixmap, img_w: int, img_h: int):
        """兼容你原来的调用方式"""
        self._pix = pixmap
        self._img_w = int(img_w)
        self._img_h = int(img_h)
        self.reset_view_fit()
        self.update()

    def set_image_bgr(self, bgr: np.ndarray):
        """可选：如果你想直接传 numpy BGR"""
        if bgr is None or bgr.size == 0:
            self._pix = None
            self.update()
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.set_image_pixmap(QPixmap.fromImage(qimg), w, h)

    def reset_view_fit(self):
        """把图像 fit 到窗口并居中"""
        if self._pix is None or self._img_w <= 0 or self._img_h <= 0:
            self._scale = 1.0
            self._offset = QPointF(0.0, 0.0)
            return

        vw, vh = max(1, self.width()), max(1, self.height())
        sx = vw / float(self._img_w)
        sy = vh / float(self._img_h)
        self._scale = min(sx, sy)
        self._scale = max(self._min_scale, min(self._scale, self._max_scale))

        draw_w = self._img_w * self._scale
        draw_h = self._img_h * self._scale
        self._offset = QPointF((vw - draw_w) * 0.5, (vh - draw_h) * 0.5)

    def map_to_image(self, widget_pos: QPointF, clamp=True) -> QPointF:
        """窗口坐标 -> 原图像素坐标(float)"""
        if self._pix is None or self._scale == 0:
            return QPointF(-1, -1)
        x = (widget_pos.x() - self._offset.x()) / self._scale
        y = (widget_pos.y() - self._offset.y()) / self._scale
        if clamp and self._img_w > 0 and self._img_h > 0:
            x = max(0.0, min(float(self._img_w - 1), x))
            y = max(0.0, min(float(self._img_h - 1), y))
        return QPointF(x, y)

    # ---------- QWidget events ----------
    def resizeEvent(self, event):
        # 窗口变大变小后保持“相对合理”：最简单是重新fit（你也可以改成保持当前视图）
        self.reset_view_fit()
        super().resizeEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)

        if self._pix is None:
            # 没图时画提示文字
            if self._hint_text:
                p.setPen(Qt.white)
                p.drawText(self.rect(), Qt.AlignCenter, self._hint_text)
            return

        p.setRenderHint(QPainter.SmoothPixmapTransform, True)

        draw_w = self._img_w * self._scale
        draw_h = self._img_h * self._scale
        target = QRectF(self._offset.x(), self._offset.y(), draw_w, draw_h)
        source = QRectF(0, 0, self._img_w, self._img_h)
        p.drawPixmap(target, self._pix, source)

    def wheelEvent(self, event):
        if self._pix is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return

        zoom = 1.15 if delta > 0 else 1.0 / 1.15

        old_scale = self._scale
        new_scale = old_scale * zoom
        new_scale = max(self._min_scale, min(new_scale, self._max_scale))
        if abs(new_scale - old_scale) < 1e-12:
            return

        mouse_pos = QPointF(event.pos())
        img_pt = self.map_to_image(mouse_pos, clamp=False)  # 缩放中心对应的图像坐标

        self._scale = new_scale
        # offset = mouse - img*scale
        self._offset = QPointF(mouse_pos.x() - img_pt.x() * self._scale,
                               mouse_pos.y() - img_pt.y() * self._scale)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._press_pos = QPointF(event.pos())
            self._last_mouse = QPointF(event.pos())
            self._moved = False

    def mouseMoveEvent(self, event):
        if self._dragging:
            cur = QPointF(event.pos())
            d = cur - self._last_mouse
            self._offset += d
            self._last_mouse = cur
            if (cur - self._press_pos).manhattanLength() > self._move_thresh:
                self._moved = True
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            # 如果没发生拖拽，认为是点击
            if not self._moved:
                img_pt = self.map_to_image(QPointF(event.pos()), clamp=True)
                u = int(round(img_pt.x()))
                v = int(round(img_pt.y()))
                self.clicked.emit(u, v)
