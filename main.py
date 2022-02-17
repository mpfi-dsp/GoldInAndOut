# views
import logging
import traceback
import pandas as pd
from globals import WORKFLOWS, NAV_ICON, DEFAULT_OUTPUT_DIR, VERSION_NUMBER
from views.home import HomePage
from typings import Unit, OutputOptions
from utils import pixels_conversion, unit_to_enum, to_coord_list
from views.logger import Logger
from views.workflow import WorkflowPage
# stylesheet
from styles.stylesheet import styles
# resources
import resources
# pyQT5
import PyQt5
from PyQt5.QtCore import Qt, QSize, QObject, pyqtSignal, QThread #, pyqt5_enable_new_onexit_scheme
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import (QWidget, QListWidget, QStackedWidget, QHBoxLayout, QListWidgetItem, QApplication,
                             QMainWindow)
import pandas._libs.tslibs.base
# general
from threads import DataLoadWorker
from functools import partial
import numexpr
import pathlib
import sys

try:
    # necessary since also targeting Mac/Linux
    from PyQt5.QtWinExtras import QtWin
    appId = 'goldinguy.mpfi.goldinandout'
    QtWin.setCurrentProcessExplicitAppUserModelID(appId)
except ImportError:
    pass

# try:
#     # fix crashing on exit on MACOS
#     pyqt5_enable_new_onexit_scheme(True)
# except Exception as e:
#     pass


class GoldInAndOut(QWidget):
    """ PARENT WINDOW INITIALIZATION """
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'GoldInAndOut {VERSION_NUMBER}')
        self.setWindowIcon(QIcon(':/icons/logo.ico'))
        # self.setWindowIcon(QIcon('./logo.png'))
        self.setMinimumSize(QSize(800, 850))
        self.logger_shown = False
        logging.info("Booting up...")
        # set max threads
        numexpr.set_num_threads(numexpr.detect_number_of_cores())
        logging.info("Detected %s cores...", str(numexpr.detect_number_of_cores()))
        # layout with list on left and stacked widget on right
        layout = QHBoxLayout(self, spacing=0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.nav_list = QListWidget(self)
        layout.addWidget(self.nav_list)
        self.page_stack = QStackedWidget(self)
        layout.addWidget(self.page_stack)
        # add main page
        self.home_page = HomePage(start=self.init_workflows)
        logging.info("Building layout...")
        # init ui
        self.init_ui()

    def init_ui(self):
        """ INITIALIZE MAIN CHILD WINDOW """
        logging.info("Initializing main window...")
        self.nav_list.currentRowChanged.connect(self.page_stack.setCurrentIndex)
        self.nav_list.setFrameShape(QListWidget.NoFrame)
        self.nav_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav_list.setCursor(QCursor(Qt.PointingHandCursor))
        # add main page to nav
        item = QListWidgetItem(NAV_ICON, str("MAIN"), self.nav_list)
        item.setSizeHint(QSize(60, 60))
        item.setTextAlignment(Qt.AlignCenter)
        # add each page to parent window stack
        self.page_stack.addWidget(self.home_page)
        # select first page by default
        self.nav_list.item(0).setSelected(True)
        self.home_page.show_logs_btn.clicked.connect(self.open_logger)
        # init logger
        self.dlg = Logger()

    def on_run_complete(self):
        self.home_page.start_btn.setText("Run Again")
        self.home_page.start_btn.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px; margin-top: 10px; margin-right: 450px; color: white; border-radius: 7px; background: #E89C12")
        self.home_page.progress.setValue(100)
        self.home_page.prog_animation.stop()
        
        for prop in self.home_props:
            prop.setEnabled(True)

    def open_logger(self):
        if self.logger_shown == False:
            self.home_page.show_logs_btn.setText("Hide Logger")
            self.logger_shown = True
            self.dlg.show()
        else:
            self.logger_shown = False
            self.home_page.show_logs_btn.setText("Display Logger")
            self.dlg.hide()

    def props_checked(self):
        for wf_cb in self.home_page.workflow_cbs:
            if wf_cb.isChecked():
                return True
        return False

    def init_workflows(self):
        try:
            """ INITIALIZE CHILD WORKFLOW WINDOWS """
            if len(self.home_page.img_le.text()) > 0 and len(self.home_page.csv_le.text()) > 0 and (self.props_checked() == True):
                # gui elements to disable when running
                self.home_props = [self.home_page.start_btn,
                                   self.home_page.img_le,  self.home_page.mask_le, self.home_page.csv_le, self.home_page.csv2_le, self.home_page.ip_scalar_type, self.home_page.op_scalar_type, self.home_page.output_dir_le, self.home_page.dod_cb, self.home_page.csvs_lb_i, self.home_page.csvs_ip_o, self.home_page.clust_area, self.home_page.show_logs_btn]
                for prop in self.home_props:
                    prop.setEnabled(False)
                self.home_page.start_btn.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px; margin-top: 10px; margin-right: 450px; color: white; border-radius: 7px; background: #ddd")
                self.empty_stack()
                self.home_page.progress.setValue(0)
                self.load_data()
        except Exception as e:
            print(e, traceback.format_exc())

    def on_loaded_data(self, loaded_data: list):
        try:
            self.COORDS, self.ALT_COORDS = loaded_data
            # file paths
            img_path: str = self.home_page.img_le.text() 
            mask_path: str = self.home_page.mask_le.text() 
            csv_path: str = self.home_page.csv_le.text() 
            csv2_path: str = self.home_page.csv2_le.text() 
            # output unit options
            ou: Unit = unit_to_enum(self.home_page.op_scalar_type.currentText() if self.home_page.op_scalar_type.currentText(
            ) is not None else self.home_page.ip_scalar_type.currentText() if '(in&out)' in self.home_page.csvs_lb_i.text() else 'px')
            s_o: float = float(self.home_page.csvs_ip_i.text() if '(in&out)' in self.home_page.csvs_lb_i.text(
            ) else self.home_page.csvs_ip_o.text() if len(self.home_page.csvs_ip_o.text()) > 0 else 1)
            # print("OUTPUT SCALAR", s_o)
            dod: bool = self.home_page.dod_cb.isChecked()
            o_dir: str = self.home_page.output_dir_le.text() if len(self.home_page.output_dir_le.text()) > 0 else DEFAULT_OUTPUT_DIR
            output_ops: OutputOptions = OutputOptions(output_unit=ou, output_dir=o_dir, output_scalar=s_o, delete_old=dod)
            c_area = self.home_page.clust_area.isChecked()

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
                    item = QListWidgetItem(NAV_ICON, str(WORKFLOWS[i]['name']), self.nav_list)
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
                                     clust_area=c_area,
                                     log=self.dlg
                                     ))
        except Exception as e:
            print(e, traceback.format_exc())

    def load_data(self):
        """ LOAD AND SCALE DATA """
        try:
            # edit in production
            logging.info("Loading data...")
            img_path: str = self.home_page.img_le.text() 
            mask_path: str = self.home_page.mask_le.text() 
            csv_path: str = self.home_page.csv_le.text() 
            csv2_path: str = self.home_page.csv2_le.text() 
            unit = unit_to_enum(self.home_page.ip_scalar_type.currentText()) if self.home_page.ip_scalar_type.currentText() else Unit.PIXEL
            scalar = float(self.home_page.csvs_ip_i.text() if len(self.home_page.csvs_ip_i.text()) > 0 else 1)
            # load in data in thread
            self.load_thread = QThread()
            self.load_worker = DataLoadWorker()
            self.load_worker.moveToThread(self.load_thread)
            self.load_thread.started.connect(partial(self.load_worker.run, img_path, mask_path, csv_path, csv2_path, unit, scalar))
            self.load_worker.finished.connect(self.on_loaded_data)
            self.load_worker.finished.connect(self.load_thread.quit)
            self.load_worker.finished.connect(self.load_worker.deleteLater)
            self.load_thread.finished.connect(self.load_thread.deleteLater)
            self.load_thread.start()
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
            logging.info("Clearing old run pages...")
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
