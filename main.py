from random import randint
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QWidget, QListWidget, QStackedWidget,
                             QHBoxLayout, QListWidgetItem, QLabel, QApplication)
import sys

from home import HomePage
from macro import MacroPage
from utils import pixels_conversion

PAGE_NAMES = ["Main", "NND", "Foo", "Bar", "Baz", "Bop"]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('../gui/logo.jpg'))
        self.setMinimumSize(QSize(900, 900))
        # self.setMaximumSize(QSize(900, 900))
        self.setWindowTitle('MPFI EM Core Pipeline')
        self.resize(900, 900)
        # layout with list on left and stacked widget on right
        layout = QHBoxLayout(self, spacing=0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.nav_list = QListWidget(self)
        layout.addWidget(self.nav_list)
        self.page_stack = QStackedWidget(self)
        layout.addWidget(self.page_stack)

        self.home_page = HomePage(start=self.start)

        self.init_ui()

    def init_ui(self):
        # create interface, hide scrollbar
        self.nav_list.currentRowChanged.connect(
            self.page_stack.setCurrentIndex)
        self.nav_list.setFrameShape(QListWidget.NoFrame)
        self.nav_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        item = QListWidgetItem(
            QIcon('foo.jpg'), str("Main"), self.nav_list)
        item.setSizeHint(QSize(60, 60))
        item.setTextAlignment(Qt.AlignCenter)

        self.page_stack.addWidget(self.home_page)
        # select first page by default
        self.nav_list.item(0).setSelected(True)

    def init_macros(self):
        # add page tabs
        for i in range(5):
            if i > 0:
                item = QListWidgetItem(
                    QIcon('foo.jpg'), str(PAGE_NAMES[i]), self.nav_list)
                item.setSizeHint(QSize(60, 60))
                item.setTextAlignment(Qt.AlignCenter)

        knn_page = MacroPage(header_name="N Nearest Distance", desc="Find the nearest distance between gold particles. Optionally generate random coordinates.", img_dropdown=[self.home_page.img_le.text()], mask_dropdown=[self.home_page.mask_le.text()], csv_dropdown=[self.home_page.csv_le.text()], parameters=["rand_particles"])
        self.page_stack.addWidget(knn_page)

        for i in range(5):
            if i > 1:
                label = QLabel(f'{PAGE_NAMES[i]}Page', self)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet('background: rgb(%d, %d, %d); margin: 50px;' % (
                    randint(0, 255), randint(0, 255), randint(0, 255)))
                self.page_stack.addWidget(label)

    def start(self):
        self.init_macros()


styles = """
QListWidget, QListView, QTreeWidget, QTreeView {
    outline: 0px;
}
QListWidget {
    min-width: 120px;
    max-width: 120px;
    color: white;
    background: teal;
    font-weight: 500;
    font-size: 18px;
}
QListWidget::item:selected {
    background: rgb(16,100,112);
    border-left: 3px solid #01D4B4;
    color: white;
}
HistoryPanel::item:hover {background: rgb(52, 52, 52);}


/* QStackedWidget {background: rgb(30, 30, 30);} */


QCheckBox {
    margin-right: 50px;
    spacing: 5px;
    font-size: 18px;    
}

QCheckBox::indicator {
    width:  27px;
    height: 27px;
}

QProgressBar {
text-align: center;
border: solid grey;
border-radius: 7px;
color: black;
background: #ddd;
font-size: 20px;
}
QProgressBar::chunk {
background-color: #05B8CC;
border-radius :7px;
}      

QPushButton {
font-size: 16px; 
font-weight: 600; 
padding: 8px; 
background: teal; 
color: white; 
border-radius: 7px;
}

QLineEdit {
font-size: 16px; 
padding: 8px; 
font-weight: 400; 
background: #ddd; 
border-radius: 7px; 
margin-bottom: 5px;
}

QComboBox {
font-size: 16px; 
padding: 8px; 
font-weight: 400; 
background: #ddd; 
border-radius: 7px; 
margin-bottom: 5px;
}

QComboBox QAbstractItemView {
font-size: 16px; 
padding: 2px;
border: 0 !important; 
outline: none !important; 
color: teal;
font-weight: 400; 
background: #ccc; 
border-radius: 7px; 
margin-bottom: 5px;
}

QLabel {
font-size: 20px; 
font-weight: bold; 
padding-top: 15px; 
padding-bottom: 10px;
color: black;
}
"""

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(styles)
    app.setStyle("fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
