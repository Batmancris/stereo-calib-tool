# -*- coding: utf-8 -*-
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


def build_app_stylesheet() -> str:
    return """
    QWidget {
        background: #08131a;
        color: #d9edf7;
        font-family: "Microsoft YaHei UI";
        font-size: 13px;
    }

    QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {
        background: #08131a;
    }

    QTabWidget::pane {
        border: 1px solid #173847;
        border-radius: 18px;
        background: #0b1821;
        top: -1px;
    }

    QTabBar::tab {
        background: #0d2230;
        color: #7fb9d4;
        padding: 12px 20px;
        margin-right: 6px;
        border-top-left-radius: 12px;
        border-top-right-radius: 12px;
        min-width: 116px;
        font-weight: 700;
    }

    QTabBar::tab:selected {
        background: #112b39;
        color: #ecfbff;
        border: 1px solid #1ec8d6;
        border-bottom: 0;
    }

    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0e7287, stop:1 #18c6c8);
        color: #041015;
        border: 1px solid #35e4df;
        border-radius: 12px;
        padding: 10px 16px;
        font-weight: 700;
    }

    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1391aa, stop:1 #4ef1e3);
    }

    QPushButton:pressed {
        background: #0a5d6c;
        color: #dcfbff;
    }

    QPushButton:disabled {
        background: #294651;
        color: #85a4af;
        border: 1px solid #34515c;
    }

    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background: #0d1d27;
        color: #d9edf7;
        border: 1px solid #1f4758;
        border-radius: 12px;
        padding: 8px 10px;
        selection-background-color: #1ec8d6;
        selection-color: #041015;
    }

    QLabel#ImagePanel, ClickableImageLabel {
        background: #050d12;
        border: 1px solid #1a4f66;
        border-radius: 18px;
        color: #dffcff;
    }

    QTextEdit#LogPanel {
        background: #09151d;
        border: 1px solid #173847;
        border-radius: 16px;
        color: #bfe6f4;
    }

    QGroupBox {
        background: #0a1720;
        border: 1px solid #173847;
        border-radius: 18px;
        margin-top: 14px;
        padding: 16px 14px 14px 14px;
        font-weight: 700;
        color: #dbf9ff;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 14px;
        padding: 0 8px;
        color: #56e3e0;
    }

    QScrollBar:vertical {
        background: transparent;
        width: 12px;
        margin: 4px;
    }

    QScrollBar::handle:vertical {
        background: #1d5367;
        border-radius: 6px;
        min-height: 24px;
    }
    """


def create_page_header(title: str, subtitle: str, accent: str = "#18c6c8") -> QWidget:
    card = QFrame()
    card.setObjectName("PageHeader")
    card.setStyleSheet(
        f"""
        QFrame#PageHeader {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #091821,
                stop:0.55 #0d2430,
                stop:1 {accent}
            );
            border: 1px solid #184053;
            border-radius: 24px;
        }}
        QLabel#HeaderTitle {{
            background: transparent;
            color: #ecfbff;
            font-family: "Microsoft YaHei UI";
            font-size: 28px;
            font-weight: 800;
        }}
        QLabel#HeaderSubtitle {{
            background: transparent;
            color: #b7deec;
            font-family: "Microsoft YaHei UI";
            font-size: 13px;
        }}
        """
    )
    lay = QVBoxLayout(card)
    lay.setContentsMargins(24, 22, 24, 22)
    lay.setSpacing(6)

    title_label = QLabel(title)
    title_label.setObjectName("HeaderTitle")
    title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    lay.addWidget(title_label)

    subtitle_label = QLabel(subtitle)
    subtitle_label.setObjectName("HeaderSubtitle")
    subtitle_label.setWordWrap(True)
    lay.addWidget(subtitle_label)
    return card
