"""Clean, high-contrast GeoDubber interface theme."""

GOLD = "#F4B936"
BG = "#10141C"
PANEL = "#1A202B"
PANEL_BORDER = "#354154"
TEXT = "#F3F6FC"
MUTED = "#B8C3D3"

STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Sylfaen", sans-serif;
    font-size: 13px;
    color: {TEXT};
}}

QMainWindow, QDialog, QWidget#root {{
    background: {BG};
}}

QScrollArea#mainScroll {{
    background: {BG};
    border: none;
}}

QScrollArea#mainScroll QWidget#root {{
    background: {BG};
}}

QFrame#panel {{
    background: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-radius: 12px;
}}

QLabel {{
    color: {TEXT};
    background: transparent;
}}

QLabel#title {{
    color: {GOLD};
    font-size: 25px;
    font-weight: 800;
    letter-spacing: 1px;
}}

QLabel#sectionTitle {{
    color: {GOLD};
    font-size: 14px;
    font-weight: 800;
    padding-bottom: 3px;
}}

QLabel#muted {{
    color: {MUTED};
}}

QLabel#warn {{
    color: #FFD09A;
    font-weight: 700;
}}

QPushButton {{
    background: #303B4E;
    color: #FFFFFF;
    border: 1px solid #5A6A82;
    border-radius: 8px;
    padding: 8px 13px;
    min-height: 25px;
    font-size: 13px;
    font-weight: 700;
}}

QPushButton:hover {{
    background: #3C4A60;
    color: #FFFFFF;
    border: 1px solid {GOLD};
}}

QPushButton:pressed {{
    background: #252E3C;
    color: #FFFFFF;
    border: 1px solid #FFD36A;
}}

QPushButton:disabled {{
    background: #343D4B;
    color: #E0E6EF;
    border: 1px solid #65748A;
}}

QPushButton#primary {{
    background: {GOLD};
    color: #17130A;
    border: 1px solid #FFE19A;
    border-radius: 9px;
    padding: 9px 18px;
    min-height: 30px;
    font-size: 15px;
    font-weight: 800;
}}

QPushButton#primary:hover {{
    background: #FFD05C;
    color: #17130A;
    border: 1px solid #FFF0C4;
}}

QPushButton#primary:pressed {{
    background: #DFA421;
    color: #17130A;
}}

QPushButton#primary:disabled {{
    background: #92743A;
    color: #FFFFFF;
    border: 1px solid #C4A45E;
}}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
QPlainTextEdit, QTextEdit {{
    background: #252D3B;
    color: #F7F9FD;
    border: 1px solid #46546A;
    border-radius: 7px;
    padding: 5px 9px;
    min-height: 25px;
    selection-background-color: {GOLD};
    selection-color: #17130A;
}}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {GOLD};
}}

QLineEdit::placeholder {{
    color: #AEB9CA;
}}

QComboBox {{
    padding-right: 25px;
}}

QComboBox::drop-down {{
    width: 23px;
    border-left: 1px solid #46546A;
    background: #303B4E;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}}

QComboBox QAbstractItemView {{
    background: #252D3B;
    color: #F7F9FD;
    border: 1px solid #5A6A82;
    selection-background-color: {GOLD};
    selection-color: #17130A;
    padding: 4px;
    outline: none;
}}

QCheckBox {{
    color: #F0F3F9;
    spacing: 9px;
    padding: 1px 2px;
    min-height: 24px;
}}

QCheckBox::indicator {{
    width: 17px;
    height: 17px;
    border: 1px solid #8190A6;
    border-radius: 4px;
    background: #252D3B;
}}

QCheckBox::indicator:hover {{
    border: 1px solid {GOLD};
}}

QCheckBox::indicator:checked {{
    background: {GOLD};
    border: 1px solid #FFE19A;
}}

QProgressBar {{
    background: #252D3B;
    color: #FFFFFF;
    border: 1px solid #46546A;
    border-radius: 7px;
    text-align: center;
    height: 19px;
    font-weight: 700;
}}

QProgressBar::chunk {{
    background: {GOLD};
    border-radius: 6px;
}}

QPlainTextEdit, QTextEdit {{
    background: #151A23;
    color: #E5EBF4;
    border: 1px solid #39465A;
    border-radius: 8px;
    selection-background-color: {GOLD};
    selection-color: #17130A;
}}

QTableWidget {{
    background: #252D3B;
    color: {TEXT};
    gridline-color: #3B485C;
    border: 1px solid #46546A;
    border-radius: 8px;
}}

QHeaderView::section {{
    background: #202938;
    color: {GOLD};
    border: none;
    border-bottom: 1px solid #46546A;
    padding: 8px;
    font-weight: 700;
}}

QTabWidget::pane {{
    border: 1px solid #3B485C;
    border-radius: 8px;
}}

QTabBar::tab {{
    background: #252D3B;
    color: #DDE5F0;
    padding: 9px 16px;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    margin-right: 3px;
}}

QTabBar::tab:selected {{
    background: #354156;
    color: {GOLD};
    font-weight: 800;
}}

QScrollBar:vertical {{
    background: #151A23;
    width: 13px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    background: #56647A;
    border-radius: 5px;
    min-height: 35px;
}}

QScrollBar::handle:vertical:hover {{
    background: #73839C;
}}

QScrollBar:horizontal {{
    background: #151A23;
    height: 13px;
    margin: 2px;
}}

QScrollBar::handle:horizontal {{
    background: #56647A;
    border-radius: 5px;
    min-width: 35px;
}}

QListWidget, QListView {{
    background: #252D3B;
    color: #F3F6FC;
    border: 1px solid #5A6A82;
    border-radius: 8px;
    padding: 4px;
    font-size: 14px;
    outline: none;
}}

QListWidget::item {{
    padding: 10px 12px;
    margin: 2px 0;
    border: 1px solid #3B485C;
    border-radius: 7px;
    background: #2C3547;
}}

QListWidget::item:hover {{
    background: #38455B;
    border: 1px solid {GOLD};
}}

QListWidget::item:selected {{
    background: #4A3B12;
    color: #FFFFFF;
    border: 2px solid {GOLD};
}}

QDialog QPushButton#primary {{
    min-width: 140px;
}}

QStatusBar {{
    background: #171D27;
    color: #DFE6F0;
    border-top: 1px solid #354154;
}}

QToolTip {{
    background: #303B4E;
    color: #FFFFFF;
    border: 1px solid {GOLD};
    padding: 6px 9px;
}}
"""
