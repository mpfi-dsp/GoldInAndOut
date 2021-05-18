# pyQT5
from PyQt5.QtWidgets import (QLabel, QFileDialog, QSpacerItem, QCheckBox, QHBoxLayout, QPushButton, QWidget,
                             QSizePolicy, QFormLayout, QLineEdit, QColorDialog, QComboBox, QProgressBar)
# general
from pathlib import Path
from functools import partial
from colorthief import ColorThief
# utils
from globals import UNIT_OPS
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
        self.header = QLabel(HEADER)
        self.header.setStyleSheet("font-size: 24px; font-weight: bold; padding-top: 8px; ")
        layout.addRow(self.header)
        self.desc = QLabel(DESC)
        self.desc.setStyleSheet("font-size: 17px; font-weight: 400; padding-top: 3px; padding-bottom: 20px;")
        self.desc.setWordWrap(True)
        layout.addRow(self.desc)

        # upload header
        self.upload_header = QLabel("Upload Files")
        layout.addRow(self.upload_header)

        # img btn
        self.img_btn = QPushButton('Upload Image', self)
        self.img_btn.clicked.connect(partial(self.open_file_picker, "img"))
        # img input
        self.img_le = QLineEdit()
        self.img_le.setPlaceholderText("None Selected")
        # add img row
        layout.addRow(self.img_btn, self.img_le)

        # mask btn
        self.mask_btn = QPushButton('Upload Mask', self)
        self.mask_btn.clicked.connect(partial(self.open_file_picker, "mask"))
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
        self.csv_btn.clicked.connect(partial(self.open_file_picker, "csv"))
        # csv input
        self.csv_le = QLineEdit()
        self.csv_le.setPlaceholderText("None Selected")
        # add csv row
        layout.addRow(self.csv_btn, self.csv_le)

        # csv2 btn
        self.csv2_btn = QPushButton('Upload CSV2', self)
        self.csv2_btn.clicked.connect(partial(self.open_file_picker, "csv2"))
        # csv2 input
        self.csv2_le = QLineEdit()
        self.csv2_le.setPlaceholderText("None Selected")
        # add csv2 row
        layout.addRow(self.csv2_btn, self.csv2_le)

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)

        # workflows header
        self.workflows_header = QLabel("Select Workflows")
        layout.addRow(self.workflows_header)

        # workflows
        self.annotate_particles_cb = QCheckBox("Workflow 1")

        self.nnd_cb = QCheckBox("Nearest Neighbor Distance")
        self.nnd_cb.setChecked(True)
        # TODO: enable other workflows
        self.calc_dens_cb = QCheckBox("Workflow 3")

        self.output_files_cb = QCheckBox("Workflow 4")

        layout.addRow(self.annotate_particles_cb, self.nnd_cb)
        layout.addRow(self.calc_dens_cb, self.output_files_cb)

        layout.addItem(spacer)

        # props header
        self.props_header = QLabel("Global Parameters")
        layout.addRow(self.props_header)


        self.ip_scalr_lb = QLabel("Input Unit")
        self.ip_scalr_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.ip_scalar_type = QComboBox()
        self.ip_scalar_type.addItems(UNIT_OPS)
        self.op_scalr_lb = QLabel("Output Unit")
        self.op_scalr_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.op_scalar_type = QComboBox()
        self.op_scalar_type.addItems(UNIT_OPS)
        self.csvs_lb = QLabel("Scalar (Pixel to Metric)")
        self.csvs_lb.setStyleSheet("font-size: 17px; font-weight: 400; margin-left: 5px;")
        self.csvs_ip = QLineEdit()
        self.csvs_ip.setStyleSheet(
            "font-size: 16px; padding: 8px;  font-weight: 400; background: #ddd; border-radius: 7px;  margin-bottom: 5px; max-width: 75px;")
        self.csvs_ip.setPlaceholderText("1")
        glob_props = QHBoxLayout()
        glob_props.addWidget(self.ip_scalr_lb)
        glob_props.addWidget(self.ip_scalar_type)
        glob_props.addWidget(self.op_scalr_lb)
        glob_props.addWidget(self.op_scalar_type)
        glob_props.addWidget(self.csvs_lb)
        glob_props.addWidget(self.csvs_ip)
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
        layout.addRow(self.start_btn)

        # assign layout
        self.setLayout(layout)

    # open file picker and print file names
    def open_file_picker(self, btn_type):
        root_dir = str(Path.home())
        file = QFileDialog.getOpenFileName(self, 'Open file', root_dir)
        filename = file[0]
        if (len(filename)) > 0:
            if btn_type == "img":
                self.img_le.setText(filename)
            elif btn_type == "mask":
                try:
                    self.mask_le.setText(filename)
                    # get dominant color in layer mask and assign to btn bg
                    palette = ColorThief(filename)
                    (r, g, b) = palette.get_color(quality=1)
                    def clamp(x):
                        return max(0, min(x, 255))
                    hex = "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))
                    comp_color = get_complimentary_color(hex)
                    self.clr_btn.setStyleSheet( f'QWidget {{background-color: {hex}; font-size: 16px; font-weight: 600; padding: 8px; color: {comp_color}; border-radius: 7px; }}')
                    self.clr_btn.setText(hex)
                except Exception as e:
                    print(e)
            elif btn_type == "csv":
                self.csv_le.setText(filename)
            elif btn_type == "csv2":
                self.csv2_le.setText(filename)

    def set_mask_clr(self):
        color = QColorDialog.getColor().name(0)
        print(color)
        comp_color = get_complimentary_color(color)
        self.clr_btn.setStyleSheet(
            f'QWidget {{background-color: {color}; font-size: 16px; font-weight: 600; padding: 8px; color: {comp_color}; border-radius: 7px; }}')
        self.clr_btn.setText(color)

