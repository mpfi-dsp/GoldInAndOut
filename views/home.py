# pyQT5
import os
import traceback

import cv2
from PyQt5.QtGui import QCursor, QMovie, QPixmap, QImage
from PyQt5.QtWidgets import (QLabel, QFileDialog, QSpacerItem, QCheckBox, QHBoxLayout, QPushButton, QWidget,
                             QSizePolicy, QFormLayout, QLineEdit, QColorDialog, QComboBox, QProgressBar, QVBoxLayout)
from PyQt5.QtCore import Qt, QByteArray, QPropertyAnimation, QAbstractAnimation, QVariantAnimation
# general
from pathlib import Path
from functools import partial
# utils
from globals import UNIT_OPS, WORKFLOWS, MAX_DIRS_PRUNE, UNIT_PX_SCALARS, DEFAULT_OUTPUT_DIR, PROG_COLOR_1, PROG_COLOR_2
from typings import FileType
from utils import get_complimentary_color

HEADER = "Gold Cluster Analysis For Freeze Fracture"
DESC = "Simply upload the appropriate files, check the workflows you'd like to run, and click \"Start\"!"

class HomePage(QWidget):
    """
    MAIN PAGE
    ________________
    @start: begins running selected workflows and display all subpages
    """
    def __init__(self, start: partial):
        super().__init__()
        # init layout
        layout = QFormLayout()
        # header
        header = QLabel(HEADER)
        header.setStyleSheet("font-size: 24px; font-weight: bold; padding-top: 8px; ")
        layout.addRow(header)
        desc = QLabel(DESC)
        desc.setStyleSheet("font-size: 17px; font-weight: 400; padding-top: 3px; padding-bottom: 10px;")
        desc.setWordWrap(True)
        layout.addRow(desc)
        # upload header
        self.upload_header = QLabel("Upload Files")
        layout.addRow(self.upload_header)
        # img btn
        img_btn = QPushButton('Upload Image', self)
        img_btn.setCursor(QCursor(Qt.PointingHandCursor))
        img_btn.clicked.connect(partial(self.open_file_picker, FileType.IMAGE))
        # img input
        self.img_le = QLineEdit()
        self.img_le.setPlaceholderText("None Selected") 
        # add img row
        layout.addRow(img_btn, self.img_le)
        # mask btn
        mask_btn = QPushButton('Upload Mask', self)
        mask_btn.setCursor(QCursor(Qt.PointingHandCursor))
        mask_btn.clicked.connect(partial(self.open_file_picker,  FileType.MASK))
        # mask input
        self.mask_le = QLineEdit()
        self.mask_le.setPlaceholderText("None Selected")
        # add mask row
        layout.addRow(mask_btn, self.mask_le)
        # csv btn
        csv_btn = QPushButton('Upload CSV', self)
        csv_btn.setCursor(QCursor(Qt.PointingHandCursor))
        csv_btn.clicked.connect(partial(self.open_file_picker, FileType.CSV))
        # csv input
        self.csv_le = QLineEdit()
        self.csv_le.setPlaceholderText("None Selected")
        # add csv row
        layout.addRow(csv_btn, self.csv_le)
        # csv2 btn
        csv2_btn = QPushButton('Upload CSV2', self)
        csv2_btn.setCursor(QCursor(Qt.PointingHandCursor))
        csv2_btn.clicked.connect(partial(self.open_file_picker, FileType.CSV2))
        # output_dir input
        self.csv2_le = QLineEdit()
        self.csv2_le.setPlaceholderText("None Selected")
        # add output
        layout.addRow(csv2_btn, self.csv2_le)
        spacer = QSpacerItem(15, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)

        # output folder header
        workflows_header = QLabel("Output Folder")
        layout.addRow(workflows_header)
        # output folder btn
        out_btn = QPushButton('Select Output', self)
        out_btn.setCursor(QCursor(Qt.PointingHandCursor))
        out_btn.clicked.connect(partial(self.open_folder_picker))
        # output folder input
        self.output_dir_le = QLineEdit()
        self.output_dir_le.setPlaceholderText(DEFAULT_OUTPUT_DIR)
        self.output_dir_le.setText(DEFAULT_OUTPUT_DIR)
        layout.addRow(out_btn, self.output_dir_le)
        layout.addItem(spacer)

        # workflows header
        workflows_header = QLabel("Select Workflows")
        layout.addRow(workflows_header)
        # workflows
        self.workflow_cbs = []
        v_cb = QVBoxLayout()
        for i in range(len(WORKFLOWS)):
            cb = QCheckBox(WORKFLOWS[i]['header'])
            if WORKFLOWS[i]['checked']:
                cb.setChecked(True)
            self.workflow_cbs.append(cb)
            v_cb.addWidget(cb)
        layout.addRow(v_cb)
        layout.addItem(spacer)
        # props header
        props_header = QLabel("Global Parameters")
        layout.addRow(props_header)
        # delete old dirs checkbox
        self.dod_cb = QCheckBox(f'prune old output (delete folders older than {MAX_DIRS_PRUNE} runs)')
        layout.addRow(self.dod_cb)
        # show logs checkbox
        self.show_logs = QCheckBox('display logger (open in new window)')
        layout.addRow(self.show_logs)
        # cluster area checkbox
        self.clust_area = QCheckBox('find cluster area')
        layout.addRow(self.clust_area)
        # input
        ip_scalr_lb = QLabel("in")
        ip_scalr_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.ip_scalar_type = QComboBox()
        self.ip_scalar_type.addItems(UNIT_OPS)
        self.ip_scalar_type.currentTextChanged.connect(self.on_input_changed)
        op_scalr_lb = QLabel("out")
        op_scalr_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.op_scalar_type = QComboBox()
        self.op_scalar_type.addItems(UNIT_OPS)
        self.op_scalar_type.currentTextChanged.connect(self.on_output_changed)
        # scalar IP to PX
        self.csvs_lb_i = QLabel("1px = __mu")  # QLabel("input to px ratio")
        self.csvs_lb_i.setStyleSheet("font-size: 17px; font-weight: 400; margin-left: 15px; ")
        self.csvs_ip_i = QLineEdit()
        self.csvs_ip_i.setStyleSheet(
            "font-size: 16px; padding: 8px;  font-weight: 400; background: #ddd; border-radius: 7px;  margin-bottom: 5px; max-width: 150px; ")
        self.csvs_ip_i.setPlaceholderText("1")

        # scalar PX to Output
        self.csvs_lb_o =  QLabel("1px = __mu") # QLabel("px to output ratio")
        self.csvs_lb_o.setStyleSheet("font-size: 17px; font-weight: 400; margin-left: 15px; ")
        self.csvs_ip_o = QLineEdit()
        self.csvs_ip_o.setStyleSheet("font-size: 16px; padding: 8px;  font-weight: 400; background: #ddd; border-radius: 7px;  margin-bottom: 5px; max-width: 150px; ")
        self.csvs_ip_o.setPlaceholderText("1")
        # hide by default if px
        self.csvs_lb_o.setHidden(True)
        self.csvs_ip_o.setHidden(True)
        self.csvs_lb_i.setHidden(True)
        self.csvs_ip_i.setHidden(True)
        # global props
        glob_props = QHBoxLayout()
        for glob in [ip_scalr_lb, self.ip_scalar_type, op_scalr_lb, self.op_scalar_type, self.csvs_lb_i, self.csvs_ip_i, self.csvs_lb_o, self.csvs_ip_o]:
        # for glob in [op_scalr_lb, self.op_scalar_type, csvs_lb, self.csvs_ip]:
            glob_props.addWidget(glob)
        layout.addRow(glob_props)
        # spacer
        layout.addItem(spacer)
        # homepage progress bar
        self.progress = QProgressBar(self)
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMaximum(100)
        layout.addRow(self.progress)
        # start btn
        self.start_btn = QPushButton('Start', self)
        self.start_btn.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 8px; margin-top: 10px; margin-right: 450px; background: #E89C12; color: white; border-radius: 7px; ")
        self.start_btn.clicked.connect(start)
        self.start_btn.setCursor(QCursor(Qt.PointingHandCursor))
        layout.addRow(self.start_btn)
        # progress bar animation
        self.prog_animation = QVariantAnimation(  # QPropertyAnimation(
            self,
            valueChanged=self._animate_prog,
            startValue=0.00001,
            endValue=0.9999,
            duration=2000
        )
        self.prog_animation.setDirection(QAbstractAnimation.Forward)
        self.prog_animation.finished.connect(
            self.prog_animation.start if self.progress.value() < 100 else self.prog_animation.stop)
        # self.prog_animation.finished.connect(self.prog_animation.deleteLater)
        self.prog_animation.start()

        # assign layout
        self.setLayout(layout)

    def _animate_prog(self, value):
        # print('prog', self.progress.value())
        if not self.start_btn.isEnabled():
            if self.progress.value() < 100:
                qss = """
                    text-align: center;
                    border: solid grey;
                    border-radius: 7px;
                    color: white;
                    font-size: 20px;
                """
                bg = "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 {color1}, stop:{value} {color2}, stop: 1.0 {color1});".format(
                    color1=PROG_COLOR_1.name(), color2=PROG_COLOR_2.name(), value=value
                )
                qss += bg
                self.progress.setStyleSheet(qss)

    def on_input_changed(self, value: str):
        if value == "px":
            self.csvs_lb_i.setHidden(True)
            self.csvs_ip_i.setHidden(True)
        else:
            self.csvs_lb_i.setHidden(False)
            self.csvs_ip_i.setHidden(False)
        self.csvs_lb_i.setText(f"(in) 1px=__{value}")
        self.csvs_ip_i.setText(str(UNIT_PX_SCALARS[value]))
        self.simplify_input(value)


    def on_output_changed(self, value):
        if value == "px":
            self.csvs_lb_o.setHidden(True)
            self.csvs_ip_o.setHidden(True)
        else:
            self.csvs_lb_o.setHidden(False)
            self.csvs_ip_o.setHidden(False)
        self.csvs_lb_o.setText(f"(out) 1px=__{value}")
        self.csvs_ip_o.setText(str(UNIT_PX_SCALARS[value]))
        self.simplify_input(value)

    def simplify_input(self, value):
        if self.ip_scalar_type.currentText() == self.op_scalar_type.currentText():
            self.csvs_lb_i.setText(f"(in&out) 1px=__{value}")
            self.csvs_lb_o.setHidden(True)
            self.csvs_ip_o.setHidden(True)
        else:
            self.csvs_lb_i.setText(
                f"(in) 1px=__{self.ip_scalar_type.currentText()}")
    def open_folder_picker(self):
        self.output_dir_le.setText(QFileDialog.getExistingDirectory(self, 'Select Output Folder'))

    def open_file_picker(self, btn_type: FileType):
        """ OPEN FILE PICKER """
        try:
            path = str(Path.home())
            if len(self.img_le.text()) > 0:
                path = os.path.dirname(self.img_le.text())
            elif len(self.mask_le.text()) > 0:
                path = os.path.dirname(self.mask_le.text())
            elif len(self.csv_le.text()) > 0:
                path = os.path.dirname(self.csv_le.text())
            elif len(self.csv2_le.text()) > 0:
                path = os.path.dirname(self.csv2_le.text())
            file = QFileDialog.getOpenFileName(self, 'Open file', path)
            filename = file[0]
            if (len(filename)) > 0:
                if btn_type == FileType.IMAGE:
                    self.img_le.setText(filename)
                elif btn_type == FileType.MASK:
                    self.mask_le.setText(filename)
                elif btn_type == FileType.CSV:
                    self.csv_le.setText(filename)
                elif btn_type == FileType.CSV2:
                    self.csv2_le.setText(filename)
        except Exception as e:
            print(e, traceback.format_exc())

    def set_mask_clr(self):
        """ MASK COLOR SET """
        color = QColorDialog.getColor().name(0)
        # print(color)
        comp_color = get_complimentary_color(color)
        self.clr_btn.setStyleSheet(
            f'QWidget {{background-color: {color}; font-size: 16px; font-weight: 600; padding: 8px; color: {comp_color}; border-radius: 7px; }}')
        self.clr_btn.setText(color)
