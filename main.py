# views
import pandas as pd

from home import HomePage
from typings import Unit
from utils import pixels_conversion
from workflow import WorkflowPage
# stylesheet
from styles.stylesheet import styles
# pyQT5
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QWidget, QListWidget, QStackedWidget, QHBoxLayout, QListWidgetItem, QLabel, QApplication)
# general
from random import randint
import sys


PAGE_NAMES = ["Main", "NND", "Foo", "Bar", "Baz", "Bop"]
CSV_SCALAR = 1

""" PARENT WINDOW INITIALIZATION """
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MPFI EM Core Pipeline')
        self.setWindowIcon(QIcon('../gui/assets/logo.jpg'))
        self.setMinimumSize(QSize(900, 950))
        self.setMaximumSize(QSize(900, 950))
        # layout with list on left and stacked widget on right
        layout = QHBoxLayout(self, spacing=0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.nav_list = QListWidget(self)
        layout.addWidget(self.nav_list)
        self.page_stack = QStackedWidget(self)
        layout.addWidget(self.page_stack)
        # add main page
        self.home_page = HomePage(start=self.init_macros)
        # init ui
        self.init_ui()

    """ INITIALIZE MAIN CHILD WINDOW """
    def init_ui(self):
        # create interface, hide scrollbar
        self.nav_list.currentRowChanged.connect(
            self.page_stack.setCurrentIndex)
        self.nav_list.setFrameShape(QListWidget.NoFrame)
        self.nav_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # add main page to nav
        item = QListWidgetItem(
            QIcon('foo.jpg'), str("Main"), self.nav_list)
        item.setSizeHint(QSize(60, 60))
        item.setTextAlignment(Qt.AlignCenter)
        # add each page to parent window stack
        self.page_stack.addWidget(self.home_page)
        # select first page by default
        self.nav_list.item(0).setSelected(True)

    """ INITIALIZE CHILD WORKFLOW WINDOWS """
    def init_macros(self):
        self.load_data()
        # add page tabs
        for i in range(5):
            if i > 0:
                item = QListWidgetItem(
                    QIcon('foo.jpg'), str(PAGE_NAMES[i]), self.nav_list)
                item.setSizeHint(QSize(60, 60))
                item.setTextAlignment(Qt.AlignCenter)

        # TODO: remove when no longer using for testing
        img_drop = self.home_page.img_le.text() if len(self.home_page.img_le.text()) > 0 else ["C:/Users/goldins/Downloads/example_image.tif"]
        mask_drop = self.home_page.mask_le.text() if len(self.home_page.mask_le.text()) > 0 else ["C:/Users/goldins/Downloads/example_mask.tif"]
        csv_drop = self.home_page.csv_le.text() if len(self.home_page.csv_le.text()) > 0 else ["C:/Users/goldins/Downloads/example_csv.csv"]
        nnd_page = WorkflowPage(scaled_df=self.SCALED_DF, header_name="Nearest Neighbor Distance", desc="Find the nearest neighbor distance between gold particles. Optionally generate random coordinates.", img_dropdown=img_drop, csv_scalar=float(self.home_page.csvs_ip.text() if len(self.home_page.csvs_ip.text()) > 0 else 1),  mask_dropdown=mask_drop, csv_dropdown=csv_drop, input_unit=self.home_page.scalar_type.currentText() if self.home_page.scalar_type.currentText() else 'px', props=["rand_particles"])
        self.page_stack.addWidget(nnd_page)
        # TODO: make each individual child window (for now random suffices)
        for i in range(5):
            if i > 1:
                label = QLabel(f'{PAGE_NAMES[i]}Page', self)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet('background: rgb(%d, %d, %d); margin: 50px;' % (
                    randint(0, 255), randint(0, 255), randint(0, 255)))
                self.page_stack.addWidget(label)

    def load_data(self):
        path = self.home_page.csv_le.text() if len(self.home_page.csv_le.text()) > 0 else "C:/Users/goldins/Downloads/example_csv.csv"
        unit = self.home_page.scalar_type.currentText() if self.home_page.scalar_type.currentText() else 'px'
        scalar = float(self.home_page.csvs_ip.text() if len(self.home_page.csvs_ip.text()) > 0 else 1)
        self.SCALED_DF = pixels_conversion(csv_path=path, input_unit=unit, csv_scalar=scalar)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(styles)
    app.setStyle("fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
