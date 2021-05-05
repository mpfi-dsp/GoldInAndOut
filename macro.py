# QT5
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QTextEdit, QAction, QFileDialog, QApplication,
                             QSpacerItem, QDialog, QRadioButton, QCheckBox, QHBoxLayout, QGraphicsColorizeEffect,
                             QPushButton, QWidget, QGridLayout, QSizePolicy, QFormLayout, QLineEdit, QColorDialog,
                             QComboBox, QProgressBar)
import time
from functools import partial

from nnd import run_nnd

class Progress(QThread):
    prog = pyqtSignal(int)

    def update_progress(self, count):
        self.prog.emit(count)

class MacroPage(QWidget):
    def __init__(self, header_name="Undefined", desc="Undefined", img_dropdown=[], mask_dropdown=[], csv_dropdown=[],
                 parameters=[]):
        super().__init__()
        self.OUTPUT_DF = pd.DataFrame()

        layout = QFormLayout()

        # header
        self.header = QLabel(header_name)
        self.header.setStyleSheet("font-size: 24px; font-weight: bold; padding-top: 8px; ")
        layout.addRow(self.header)
        self.desc = QLabel(desc)
        self.desc.setStyleSheet("font-size: 17px; font-weight: 400; padding-top: 3px; padding-bottom: 20px;")
        self.desc.setWordWrap(True)
        layout.addRow(self.desc)

        # upload header
        self.upload_header = QLabel("File Selection")
        layout.addRow(self.upload_header)

        # add dropdown fields
        # if len(img_dropdown[0]) > 0:
        self.img_lb = QLabel("Img")
        self.img_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.img_drop = QComboBox()
        self.img_drop.addItems(
            img_dropdown if len(img_dropdown[0]) > 0 else ["C:/Users/goldins/Downloads/example_image.tif"])
        layout.addRow(self.img_lb, self.img_drop)
        # if len(mask_dropdown[0]) > 0:
        self.mask_lb = QLabel("Mask")
        self.mask_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.mask_drop = QComboBox()
        self.mask_drop.addItems(mask_dropdown)
        layout.addRow(self.mask_lb, self.mask_drop)
        # if len(csv_dropdown[0]) > 0:
        self.csv_lb = QLabel("CSV")
        self.csv_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.csv_drop = QComboBox()
        self.csv_drop.addItems(
            csv_dropdown if len(csv_dropdown[0]) > 0 else ["C:/Users/goldins/Downloads/example_csv.csv"])
        layout.addRow(self.csv_lb, self.csv_drop)

        # props header
        self.workflows_header = QLabel("Parameters")
        layout.addRow(self.workflows_header)

        # props
        self.csvs_lb = QLabel("CSV Scalar")
        self.csvs_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.csvs_input = QLineEdit()
        self.csvs_input.setPlaceholderText("1")
        self.gen_rand = QLabel("Generate Rand Coords")
        self.gen_rand.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.gen_rand_cb = QCheckBox()
        param_r = QHBoxLayout()
        param_r.addWidget(self.csvs_lb)
        param_r.addWidget(self.csvs_input)
        param_r.addWidget(self.gen_rand)
        param_r.addWidget(self.gen_rand_cb)
        layout.addRow(param_r)

        self.progress = QProgressBar(self)
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMaximum(100)
        layout.addRow(self.progress)

        # run & download btns
        self.run_btn = QPushButton('Run Again', self)
        self.run_btn.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px; margin-top: auto; background: teal; color: white; border-radius: 7px; ")
        self.run_btn.clicked.connect(self.run)
        self.download_btn = QPushButton('Download', self)
        self.download_btn.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px; margin-top: auto; background: #ccc; color: white; border-radius: 7px; ")
        self.download_btn.clicked.connect(partial(self.download, "nnd_output.csv"))
        btn_r = QHBoxLayout()
        btn_r.addWidget(self.run_btn)
        btn_r.addWidget(self.download_btn)
        layout.addRow(btn_r)

        # assign layout
        self.setLayout(layout)

    def on_progress_update(self, value):
        self.progress.setValue(value)

    def run(self):
        try:
            prog_wrapper = Progress()
            prog_wrapper.prog.connect(self.on_progress_update)

            # run knn
            self.OUTPUT_DF = run_nnd(prog_wrapper=prog_wrapper, img_path=self.img_drop.currentText(), csv_path=self.csv_drop.currentText(),
                                         pface_path="",
                                         csv_scalar=(self.csvs_input.text() if len(self.csvs_input.text()) > 0 else 1),
                                         gen_rand=self.gen_rand_cb.isChecked())
            self.progress.setValue(100)
            # # update progress bar
            # nnd = NNDWrapper()
            # nnd.countChanged.connect(self.onCountChanged)
            #
            # # run knn
            # self.OUTPUT_DF = nnd.run_nnd(img_path=self.img_drop.currentText(), csv_path=self.csv_drop.currentText(),
            #                     pface_path="",
            #                     csv_scalar=(self.csvs_input.text() if len(self.csvs_input.text()) > 0 else 1),
            #                     gen_rand=self.gen_rand_cb.isChecked())
            print(self.OUTPUT_DF.head())
            self.download_btn.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px; margin-top: auto; background: teal; color: white; border-radius: 7px; ")
        except Exception as e:
            print(e)

    def download(self, file_name):
        if self.OUTPUT_DF.shape[0] > 0 and self.OUTPUT_DF.shape[1] > 0:
            try:
                self.OUTPUT_DF.to_csv(file_name, index=False, header=True)
            except Exception as e:
                print(e)
