# views
import logging
import traceback
import pandas as pd

# from confetti import SuccessGif
from globals import WORKFLOWS, NAV_ICON, DEFAULT_OUTPUT_DIR
from views.home import HomePage
from typings import Unit, OutputOptions
from utils import pixels_conversion, unit_to_enum, to_coord_list
from views.logger import Logger
from views.workflow import WorkflowPage
# stylesheet
from styles.stylesheet import styles
# pyQT5
import PyQt5
from PyQt5.QtCore import Qt, QSize, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import (QWidget, QListWidget, QStackedWidget, QHBoxLayout, QListWidgetItem, QApplication,
                             QMainWindow)
import pandas._libs.tslibs.base
# general
import sys
from functools import partial
import numexpr
from workflows.random_coords import gen_random_coordinates
import pathlib

class GoldInAndOut(QWidget):
    """ PARENT WINDOW INITIALIZATION """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GoldInAndOut')
        current_directory = str(pathlib.Path(__file__).parent.absolute())
        iconp = current_directory + '/logo.ico'
        self.setWindowIcon(QIcon(iconp))
        self.setMinimumSize(QSize(800, 1000))
        # set max threads
        numexpr.set_num_threads(numexpr.detect_number_of_cores())
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

    def init_ui(self):
        """ INITIALIZE MAIN CHILD WINDOW """
        self.nav_list.currentRowChanged.connect(self.page_stack.setCurrentIndex)
        self.nav_list.setFrameShape(QListWidget.NoFrame)
        self.nav_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav_list.setCursor(QCursor(Qt.PointingHandCursor))
        # add main page to nav
        item = QListWidgetItem(
            NAV_ICON, str("MAIN"), self.nav_list)
        item.setSizeHint(QSize(60, 60))
        item.setTextAlignment(Qt.AlignCenter)
        # add each page to parent window stack
        self.page_stack.addWidget(self.home_page)
        # select first page by default
        self.nav_list.item(0).setSelected(True)
        self.home_page.show_logs.clicked.connect(self.open_logger)
        # init logger
        self.dlg = Logger()


    def on_run_complete(self):
        self.home_page.start_btn.setText("Run Again")
        self.home_page.start_btn.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px; margin-top: 10px; margin-right: 450px; color: white; border-radius: 7px; background: #E89C12")
        self.home_page.progress.setValue(100)
        for prop in self.home_props:
            prop.setEnabled(True)

    def open_logger(self):
        if self.home_page.show_logs.isChecked():
            self.dlg.show()
        else:
            self.dlg.hide()

    def init_workflows(self):
        try:
            """ INITIALIZE CHILD WORKFLOW WINDOWS """
            self.home_props = [self.home_page.start_btn,
                               self.home_page.img_le,  self.home_page.mask_le, self.home_page.csv_le, self.home_page.csv2_le, self.home_page.ip_scalar_type, self.home_page.op_scalar_type, self.home_page.output_dir_le, self.home_page.dod_cb, self.home_page.csvs_lb_i, self.home_page.csvs_ip_o]
            for prop in self.home_props:
                prop.setEnabled(False)
            self.home_page.start_btn.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px; margin-top: 10px; margin-right: 450px; color: white; border-radius: 7px; background: #ddd")
            self.empty_stack()
            self.load_data()
            self.home_page.progress.setValue(0)

            # TODO: remove when no longer using for testing
            img_path: str = self.home_page.img_le.text() if len(self.home_page.img_le.text()) > 0 else "./input/example_image.tif"
            mask_path: str = self.home_page.mask_le.text() if len(self.home_page.mask_le.text()) > 0 else "./input/example_mask.tif"
            csv_path: str = self.home_page.csv_le.text() if len(self.home_page.csv_le.text()) > 0 else "./input/example_csv.csv"
            csv2_path: str = self.home_page.csv2_le.text() # if len(self.home_page.csv2_le.text()) > 0 else ["./input/example_csv.csv"]

            # output unit options
            ou: Unit = unit_to_enum(self.home_page.op_scalar_type.currentText() if self.home_page.op_scalar_type.currentText(
            ) is not None else self.home_page.ip_scalar_type.currentText() if '(in&out)' in self.home_page.csvs_lb_i.text else 'px')
            s_o: float = float(self.home_page.csvs_ip_o.text() if len(self.home_page.csvs_ip_o.text()) > 0 else self.home_page.csvs_ip_i.text() if '(in&out)' in self.home_page.csvs_lb_i.text else 1)
            dod: bool = self.home_page.dod_cb.isChecked()
            o_dir: str = self.home_page.output_dir_le.text() if len(self.home_page.output_dir_le.text()) > 0 else DEFAULT_OUTPUT_DIR
            output_ops: OutputOptions = OutputOptions(output_unit=ou, output_dir=o_dir, output_scalar=s_o, delete_old=dod)

            # determine workflow pages
            wf_td = 0
            for wf_cb in self.home_page.workflow_cbs:
                if wf_cb.isChecked():
                    wf_td += 1
            
            # generate workflow pages
            z = 0
            for i in range(len(WORKFLOWS)):
                if self.home_page.workflow_cbs[i].isChecked():
                    z += 1
                    item = QListWidgetItem(
                            NAV_ICON, str(WORKFLOWS[i]['name']), self.nav_list)
                    item.setSizeHint(QSize(60, 60))
                    item.setTextAlignment(Qt.AlignCenter)
                    print(WORKFLOWS[i]['name'])
                    self.page_stack.addWidget(
                        WorkflowPage(coords=self.COORDS,
                                     alt_coords=self.ALT_COORDS,
                                     wf=WORKFLOWS[i],
                                     img=img_path,
                                     mask=mask_path,
                                     csv=csv_path,
                                     csv2=csv2_path,
                                     output_ops=output_ops,
                                     pg=partial(self.update_main_progress, (int((z / wf_td * 100)))),
                                     log=self.dlg
                                     ))
        except Exception as e:
            print(e, traceback.format_exc())

    def load_data(self):
        """ LOAD AND SCALE DATA """
        path = self.home_page.csv_le.text() if len(self.home_page.csv_le.text()) > 0 else "./input/example_csv.csv"
        unit = unit_to_enum(self.home_page.ip_scalar_type.currentText()) if self.home_page.ip_scalar_type.currentText() else Unit.PIXEL
        scalar = float(self.home_page.csvs_ip_i.text() if len(self.home_page.csvs_ip_i.text()) > 0 else 1)
        try:
            data = pd.read_csv(path, sep=",")
            scaled_df = pixels_conversion(data=data, unit=unit, scalar=scalar)
            self.COORDS = to_coord_list(scaled_df)
        except Exception as e:
            print(e, traceback.format_exc())
        try:
            if len(self.home_page.csv2_le.text()) > 0:
                data = pd.read_csv(self.home_page.csv2_le.text(), sep=",")
                self.ALT_COORDS = to_coord_list(
                    pixels_conversion(data=data, unit=unit, scalar=scalar))
            else:
                # TODO: update for production
                img = self.home_page.img_le.text() if len(self.home_page.img_le.text()) > 0 else "./input/example_image.tif"
                mask = self.home_page.mask_le.text() if len(self.home_page.mask_le.text()) > 0 else "./input/example_mask.tif"
                self.ALT_COORDS = gen_random_coordinates(img, mask, count=len(self.COORDS))
        except Exception as e:
            print(e, traceback.format_exc())

    def update_main_progress(self, value: int):
        """ UPDATE PROGRESS BAR """
        if self.home_page.progress.value() != 100:
            self.home_page.progress.setValue(value)
        if value == 100:
            self.on_run_complete()

    def empty_stack(self):
        """ CLEAR PAGE/NAV STACKS """
        try:
            for i in range(self.page_stack.count()-1, 0, -1):
                if i > 0:
                    self.nav_list.takeItem(i)
                    self.page_stack.removeWidget(self.page_stack.widget(i))
        except Exception as e:
            print(e, traceback.format_exc())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(styles)
    app.setStyle("fusion")
    logging.basicConfig(level='INFO')
    gui = GoldInAndOut()
    gui.show()
    sys.exit(app.exec_())
