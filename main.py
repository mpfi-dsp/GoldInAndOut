# views
from globals import WORKFLOWS, NAV_ICON
from views.home import HomePage
from typings import Unit
from utils import pixels_conversion, unit_to_enum
from views.workflow import WorkflowPage
# stylesheet
from styles.stylesheet import styles
# pyQT5
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QWidget, QListWidget, QStackedWidget, QHBoxLayout, QListWidgetItem, QApplication)
# general
import sys


""" PARENT WINDOW INITIALIZATION """
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MPFI EM Core Pipeline')
        self.setWindowIcon(QIcon('./assets/logo.jpg'))
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
        self.home_page = HomePage(start=self.init_workflows)
        # init ui
        self.init_ui()

    """ INITIALIZE MAIN CHILD WINDOW """
    def init_ui(self):
        # create interface, hide scrollbar
        self.nav_list.currentRowChanged.connect(self.page_stack.setCurrentIndex)
        self.nav_list.setFrameShape(QListWidget.NoFrame)
        self.nav_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # add main page to nav
        item = QListWidgetItem(
            NAV_ICON, str("Main"), self.nav_list)
        item.setSizeHint(QSize(60, 60))
        item.setTextAlignment(Qt.AlignCenter)
        # add each page to parent window stack
        self.page_stack.addWidget(self.home_page)
        # select first page by default
        self.nav_list.item(0).setSelected(True)

    """ INITIALIZE CHILD WORKFLOW WINDOWS """
    def init_workflows(self):
        self.home_page.start_btn.setText("Run Again")
        self.empty_stack()
        self.load_data()

        # TODO: remove when no longer using for testing
        img_drop = self.home_page.img_le.text() if len(self.home_page.img_le.text()) > 0 else ["./input/example_image.tif"]
        mask_drop = self.home_page.mask_le.text() if len(self.home_page.mask_le.text()) > 0 else ["./input/example_mask.tif"]
        csv_drop = self.home_page.csv_le.text() if len(self.home_page.csv_le.text()) > 0 else ["./input/example_csv.csv"]
        # input/output units
        iu = unit_to_enum(self.home_page.ip_scalar_type.currentText() if self.home_page.ip_scalar_type.currentText() is not None else 'px')
        ou = unit_to_enum(self.home_page.op_scalar_type.currentText() if self.home_page.op_scalar_type.currentText() is not None else 'px')
        # scalar
        s = float(self.home_page.csvs_ip.text() if len(self.home_page.csvs_ip.text()) > 0 else 1)

        # add page tabs
        for i in range(len(WORKFLOWS)):
            self.on_progress_update(i * 20)
            item = QListWidgetItem(
                NAV_ICON, str(WORKFLOWS[i]['name']), self.nav_list)
            item.setSizeHint(QSize(60, 60))
            item.setTextAlignment(Qt.AlignCenter)
            # generate workflow page
            print(WORKFLOWS[i])
            self.page_stack.addWidget(
                WorkflowPage(scaled_df=self.SCALED_DF,
                             workflow=WORKFLOWS[i],
                             img=img_drop,
                             mask=mask_drop,
                             csv=csv_drop,
                             scalar=s,
                             input_unit=iu,
                             output_unit=ou)
            )
        self.on_progress_update(100)

    """ LOAD AND SCALE DATA """
    def load_data(self):
        path = self.home_page.csv_le.text() if len(self.home_page.csv_le.text()) > 0 else "./input/example_csv.csv"
        unit = unit_to_enum(self.home_page.ip_scalar_type.currentText()) if self.home_page.ip_scalar_type.currentText() else Unit.PIXEL
        scalar = float(self.home_page.csvs_ip.text() if len(self.home_page.csvs_ip.text()) > 0 else 1)
        self.SCALED_DF = pixels_conversion(csv_path=path, input_unit=unit, csv_scalar=scalar)

    """ UPDATE PROGRESS BAR """
    def on_progress_update(self, value):
        self.home_page.progress.setValue(value)

    """ CLEAR PAGE/NAV STACKS """
    def empty_stack(self):
        for i in range(self.page_stack.count()):
            if i > 0:
                self.nav_list.takeItem(i)
                self.page_stack.removeWidget(self.page_stack.widget(i))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(styles)
    app.setStyle("fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
