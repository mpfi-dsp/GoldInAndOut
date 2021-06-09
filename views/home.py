# pyQT5
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (QLabel, QFileDialog, QSpacerItem, QCheckBox, QHBoxLayout, QPushButton, QWidget,
                             QSizePolicy, QFormLayout, QLineEdit, QColorDialog, QComboBox, QProgressBar, QVBoxLayout)
from PyQt5.QtCore import Qt
# general
from pathlib import Path
from functools import partial
from colorthief import ColorThief
# utils
from globals import UNIT_OPS, WORKFLOWS
from typings import FileType
from utils import get_complimentary_color, pixels_conversion

HEADER = "Gold Cluster Analysis For Freeze Fracture"
DESC = "Simply upload the appropriate files, check the workflows you'd like to run, and click \"Start\"!"

""" 
MAIN PAGE 
________________
@start: begins running selected workflows and display all subpages
"""

class HomePage(QWidget):
    def __init__(self, start):
        super().__init__()
        layout = QFormLayout()

        # header
        header = QLabel(HEADER)
        header.setStyleSheet("font-size: 24px; font-weight: bold; padding-top: 8px; ")
        layout.addRow(header)
        desc = QLabel(DESC)
        desc.setStyleSheet("font-size: 17px; font-weight: 400; padding-top: 3px; padding-bottom: 20px;")
        desc.setWordWrap(True)
        layout.addRow(desc)

        # upload header
        self.upload_header = QLabel("Upload Files")
        layout.addRow(self.upload_header)

        # img btn
        self.img_btn = QPushButton('Upload Image', self)
        self.img_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.img_btn.clicked.connect(partial(self.open_file_picker, FileType.IMAGE))
        # img input
        self.img_le = QLineEdit()
        self.img_le.setPlaceholderText("None Selected")
        # add img row
        layout.addRow(self.img_btn, self.img_le)

        # mask btn
        self.mask_btn = QPushButton('Upload Mask', self)
        self.mask_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.mask_btn.clicked.connect(partial(self.open_file_picker,  FileType.MASK))
        # mask input
        self.mask_le = QLineEdit()
        self.mask_le.setPlaceholderText("None Selected")
        # TODO: mask color btn
        # self.clr_btn = QPushButton('Mask Color', self)
        # self.clr_btn.setStyleSheet("background: black;")
        # self.clr_btn.clicked.connect(self.set_mask_clr)
        # add mask row
        layout.addRow(self.mask_btn, self.mask_le)

        # csv btn
        self.csv_btn = QPushButton('Upload CSV', self)
        self.csv_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.csv_btn.clicked.connect(partial(self.open_file_picker, FileType.CSV))
        # csv input
        self.csv_le = QLineEdit()
        self.csv_le.setPlaceholderText("None Selected")
        # add csv row
        layout.addRow(self.csv_btn, self.csv_le)

        # csv2 btn
        self.csv2_btn = QPushButton('Upload CSV2', self)
        self.csv2_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.csv2_btn.clicked.connect(partial(self.open_file_picker, FileType.CSV2))
        # csv2 input
        self.csv2_le = QLineEdit()
        self.csv2_le.setPlaceholderText("None Selected")
        # add csv2 row
        layout.addRow(self.csv2_btn, self.csv2_le)

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)

        # workflows header
        workflows_header = QLabel("Select Workflows")
        layout.addRow(workflows_header)

        # workflows
        self.workflow_cbs = []
        v_cb = QVBoxLayout()
        for i in range(len(WORKFLOWS)):
            cb = QCheckBox(WORKFLOWS[i]['header'])
            cb.setChecked(True)
            self.workflow_cbs.append(cb)
            v_cb.addWidget(cb)
        layout.addRow(v_cb)
        layout.addItem(spacer)

        # props header
        props_header = QLabel("Global Parameters")
        layout.addRow(props_header)

        # TODO: Add back ability to set input unit
        # self.ip_scalr_lb = QLabel("Input Unit")
        # self.ip_scalr_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        # self.ip_scalar_type = QComboBox()
        # self.ip_scalar_type.addItems(UNIT_OPS)
        self.op_scalr_lb = QLabel("output unit")
        self.op_scalr_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.op_scalar_type = QComboBox()
        self.op_scalar_type.addItems(UNIT_OPS)
        self.csvs_lb = QLabel("scalar (px to output unit ratio)")
        self.csvs_lb.setStyleSheet("font-size: 17px; font-weight: 400; margin-left: 15px; ")
        self.csvs_ip = QLineEdit()
        self.csvs_ip.setStyleSheet("font-size: 16px; padding: 8px;  font-weight: 400; background: #ddd; border-radius: 7px;  margin-bottom: 5px; max-width: 150px; ")
        self.csvs_ip.setPlaceholderText("1")
        glob_props = QHBoxLayout()

        # for glob in [self.ip_scalr_lb, self.ip_scalar_type, self.op_scalr_lb, self.op_scalar_type, self.csvs_lb, self.csvs_ip]:
        for glob in [self.op_scalr_lb, self.op_scalar_type, self.csvs_lb, self.csvs_ip]:
            glob_props.addWidget(glob)
        layout.addRow(glob_props)

        layout.addItem(spacer)

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

        # assign layout
        self.setLayout(layout)

    """ OPEN FILE PICKER """""
    def open_file_picker(self, btn_type):
        root_dir = str(Path.home())
        file = QFileDialog.getOpenFileName(self, 'Open file', root_dir)
        filename = file[0]
        if (len(filename)) > 0:
            if btn_type == FileType.IMAGE:
                self.img_le.setText(filename)
            elif btn_type == FileType.MASK:
                try:
                    self.mask_le.setText(filename)
                    # get dominant color in layer mask and assign to btn bg
                    palette = ColorThief(filename)
                    (r, g, b) = palette.get_color(quality=1)

                    def clamp(x):
                        return max(0, min(x, 255))

                    hex = "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))
                    comp_color = get_complimentary_color(hex)
                    self.clr_btn.setStyleSheet(f'QWidget {{background-color: {hex}; font-size: 16px; font-weight: 600; padding: 8px; color: {comp_color}; border-radius: 7px; }}')
                    self.clr_btn.setText(hex)
                except Exception as e:
                    print(e)
            elif btn_type == FileType.CSV:
                self.csv_le.setText(filename)
            elif btn_type == FileType.CSV2:
                self.csv2_le.setText(filename)

    """ MASK COLOR SET """
    def set_mask_clr(self):
        color = QColorDialog.getColor().name(0)
        print(color)
        comp_color = get_complimentary_color(color)
        self.clr_btn.setStyleSheet(
            f'QWidget {{background-color: {color}; font-size: 16px; font-weight: 600; padding: 8px; color: {comp_color}; border-radius: 7px; }}')
        self.clr_btn.setText(color)
