# pyQT5
import os
import numpy as np
import traceback
import logging
import cv2
from PyQt5.QtGui import QCursor, QMovie, QPixmap, QImage, QIntValidator, QFont, QIcon
from PyQt5.QtWidgets import (QLabel, QFileDialog, QSpacerItem, QCheckBox, QHBoxLayout, QPushButton, QWidget,
                             QSizePolicy, QFormLayout, QLineEdit, QColorDialog, QComboBox, QProgressBar, QVBoxLayout, QMessageBox, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QByteArray, QPropertyAnimation, QAbstractAnimation, QVariantAnimation, QTimer, QSize
# general
from pathlib import Path
from functools import partial
# utils
from globals import UNIT_OPS, WORKFLOWS, MAX_DIRS_PRUNE, UNIT_PX_SCALARS, DEFAULT_OUTPUT_DIR, PROG_COLOR_1, PROG_COLOR_2
from typings import FileType
from utils import get_complimentary_color
from workflows import random_coords

HEADER = "Automated Gold Particle Analysis"
DESC = "Upload files, select workflows and desired parameters, and click \"Start\"!"

class HomePage(QWidget):
    """
    MAIN PAGE
    ________________
    @start: begins running selected workflows and display all subpages
    """
    def __init__(self, start: partial):
        super().__init__()
        self.folder_count = 0
        self.run_idx = 0
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
        self.upload_header = QLabel("Select Input & Output")
        # folder btn
        folder_btn = QPushButton('Upload Folder', self)
        folder_btn.setCursor(QCursor(Qt.PointingHandCursor))
        folder_btn.setToolTip('Upload all files at once. Filenames: "image" => image, "mask" => mask, "gold" => csv1, "landmark" => csv2.')
        folder_btn.setStyleSheet("max-width: 150px; ")
        folder_btn.clicked.connect(self.open_folder_picker)
        # multi-file folder btn
        self.multi_folder_btn = QPushButton('Multi-folder Upload', self)
        self.multi_folder_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.multi_folder_btn.setToolTip('Upload multiple folders at once. Filenames: "image" => image, "mask" => mask, "gold" => csv1, "landmark" => csv2, "scalar" => scalar.')
        self.multi_folder_btn.setStyleSheet("max-width: 150px; ")
        self.multi_folder_btn.clicked.connect(self.open_multi_folder_picker)
        # file btns widget
        h_bl = QHBoxLayout()
        h_bl.addWidget(self.upload_header)
        h_bl.addWidget(folder_btn)
        h_bl.addWidget(self.multi_folder_btn)
        layout.addRow(h_bl)
        # img btn
        img_btn = QPushButton('Set Image', self)
        img_btn.setCursor(QCursor(Qt.PointingHandCursor))
        img_btn.setToolTip('Supports TIF, PNG, JPG, or JPEG format..')
        img_btn.clicked.connect(partial(self.open_file_picker, FileType.IMAGE))
        # img input
        self.img_le = QLineEdit()
        self.img_le.setPlaceholderText("None Selected (TIF, PNG, JPG)") 
        # add img row
        layout.addRow(img_btn, self.img_le)
        # mask btn
        mask_btn = QPushButton('Set Mask', self)
        mask_btn.setCursor(QCursor(Qt.PointingHandCursor))
        mask_btn.setToolTip('Supports TIF, PNG, JPG, or JPEG format. Mask can be any color with white background.')
        mask_btn.clicked.connect(partial(self.open_file_picker,  FileType.MASK))
        # mask input
        self.mask_le = QLineEdit()
        self.mask_le.setPlaceholderText("None Selected (TIF, PNG, JPG)")
        # add mask row
        layout.addRow(mask_btn, self.mask_le)
        # csv btn
        csv_btn = QPushButton('Set Gold', self)
        csv_btn.setCursor(QCursor(Qt.PointingHandCursor))
        csv_btn.setToolTip('Particle population. CSV must have X and Y columns with no spaces.')
        csv_btn.clicked.connect(partial(self.open_file_picker, FileType.CSV))
        # csv input
        self.csv_le = QLineEdit()
        self.csv_le.setPlaceholderText("None Selected (CSV)")
        # add csv row
        layout.addRow(csv_btn, self.csv_le)
        # csv2 btn
        csv2_btn = QPushButton('Set Mark', self)
        csv2_btn.setCursor(QCursor(Qt.PointingHandCursor))
        csv2_btn.setToolTip('Landmark population. CSV must have X and Y columns with no spaces.')
        csv2_btn.clicked.connect(partial(self.open_file_picker, FileType.CSV2))
        # output_dir input
        self.csv2_le = QLineEdit()
        self.csv2_le.setPlaceholderText("None Selected (CSV)")
        # add output
        layout.addRow(csv2_btn, self.csv2_le)
        spacer = QSpacerItem(15, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)

        # TODO: output folder header
        # workflows_header = QLabel("Output Folder")
        # layout.addRow(workflows_header)
        # output folder btn
        out_btn = QPushButton('Set Output', self)
        out_btn.setCursor(QCursor(Qt.PointingHandCursor))
        out_btn.clicked.connect(partial(self.open_output_folder_picker))
        # output folder input
        self.output_dir_le = QLineEdit()
        self.output_dir_le.setPlaceholderText(DEFAULT_OUTPUT_DIR)
        self.output_dir_le.setText(DEFAULT_OUTPUT_DIR)
        layout.addRow(out_btn, self.output_dir_le)
        layout.addItem(spacer)
        # parameter input 
        param_btn = QPushButton('Set Parameters', self)
        param_btn.setCursor(QCursor(Qt.PointingHandCursor))
        param_btn.clicked.connect(partial(self.open_file_picker, FileType.PARAM))
        self.param_popup_btn = QPushButton('Show Example', self)
        self.param_popup_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.param_input = QLabel("Global Parameters Input Format: \nDistance=27px \nClust=2 \nRandom=1 \nSteps=10 \nSize=60px \nRadius=50px ")
        self.param_input.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.param_input.setStyleSheet("font-size: 18px; font-weight: 450;") 
        self.reload_params = QPushButton('*', self)
        self.reload_params.setToolTip('Reset Parameters')
        self.reload_params.clicked.connect(self.reload_params_func)
        #self.params_icon = QIcon('./images/reload_icon.png')
        #self.reload_params.setIcon(self.params_icon) #(QIcon('./images/reload_icon.png'))
        self.reload_params.setCursor(QCursor(Qt.PointingHandCursor))
        self.param_le = QLineEdit()
        self.param_le.setPlaceholderText("None Selected (TXT)")
        param_box = QHBoxLayout()
        param_box.addWidget(param_btn)
        param_box.addWidget(self.param_popup_btn)
        param_box.addWidget(self.reload_params)
        param_box.addWidget(self.param_le)
        layout.addRow(param_box)
        # scalar input 
        scalar_btn = QPushButton('Set Scalar', self)
        scalar_btn.setCursor(QCursor(Qt.PointingHandCursor))
        scalar_btn.clicked.connect(partial(self.open_file_picker, FileType.SCALAR))
        self.scalar_le = QLineEdit()
        self.scalar_le.setPlaceholderText("None Selected (TXT)")
        scalar_box = QHBoxLayout()
        scalar_box.addWidget(scalar_btn)
        scalar_box.addWidget(self.scalar_le)
        layout.addRow(scalar_box)
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

        props_header = QLabel("Global Parameters")
        # parameters 
        self.params_row_1 = QLabel('Distance threshold: 27px           Minimum cluster size: 2           # random trials: 1           ')
        self.params_row_1.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.params_row_2 = QLabel('Maximum # steps: 10               Step size: 60px                         Initial radius: 50px')
        self.params_row_2.setStyleSheet("font-size: 17px; font-weight: 400;")
        # folder btn
        self.show_logs_btn = QPushButton('Display Logger', self)
        self.show_logs_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.show_logs_btn.setToolTip('Open in new window')
        self.show_logs_btn.setStyleSheet("max-width: 150px; ")
        # props header
        p_bl = QHBoxLayout()
        p_bl.addWidget(props_header)
        p_bl.addWidget(self.show_logs_btn)
        layout.addRow(p_bl)
        layout.addRow(self.params_row_1)
        layout.addRow(self.params_row_2)
        # delete old dirs checkbox
        self.dod_cb = QCheckBox(f'prune old output (delete folders older than {MAX_DIRS_PRUNE} runs)')
        layout.addRow(self.dod_cb)
        # show logs checkbox
        # self.show_logs = QCheckBox('display logger (open in new window)')
        # layout.addRow(self.show_logs)
        # cluster area checkbox
        self.clust_area = QCheckBox('find cluster area')
        layout.addRow(self.clust_area)
        # automate checking of "find cluster area" for Cluster/Separation
        self.workflow_cbs[1].stateChanged.connect(self.find_clust_func)
        self.workflow_cbs[2].stateChanged.connect(self.find_clust_func)
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
        # parameters  
        self.parameters = [27, 2, 10, 60, 50] # default params
        # global props
        glob_props = QHBoxLayout()
        for glob in [ip_scalr_lb, self.ip_scalar_type, op_scalr_lb, self.op_scalar_type, self.csvs_lb_i, self.csvs_ip_i, self.csvs_lb_o, self.csvs_ip_o]:
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
            "font-size: 16px; font-weight: 600; padding: 8px; margin-top: 10px; margin-right: 5px; background: #E89C12; color: white; border-radius: 7px; ")
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

    # automate checking of "find cluster area" for Cluster/Separation
    def find_clust_func(self): 
        if self.workflow_cbs[1].isChecked() or self.workflow_cbs[2].isChecked():
            self.clust_area.setChecked(True)
        elif not self.workflow_cbs[1].isChecked() or not self.workflow_cbs[2].isChecked():
            self.clust_area.setChecked(False)

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
            if self.op_scalar_type.currentText() != "px":
                self.csvs_lb_o.setHidden(False)
                self.csvs_ip_o.setHidden(False)

    def reset_file_selection(self):
        self.multi_folder_btn.setText("Multi-folder Upload")
        self.img_le.setText("")
        self.mask_le.setText("")
        self.csv_le.setText("")
        self.csv2_le.setText("")
        self.param_le.setText("")
        self.scalar_le.setText("")
        self.parameters = [27, 2, 10, 60, 50]
        random_coords.N = 1
        self.params_row_1.repaint()
        self.params_row_1.setText(f'Distance threshold: {self.parameters[0]}px           Minimum cluster size: {self.parameters[1]}           # random trials: {random_coords.N}           ')
        self.params_row_2.repaint()
        self.params_row_2.setText(f'Maximum # steps: {self.parameters[2]}               Step size: {self.parameters[3]}px                         Initial radius: {self.parameters[4]}px')
        self.ip_scalar_type.setCurrentIndex(0)
        self.op_scalar_type.setCurrentIndex(0)
        self.csvs_ip_i.setText("")
        self.csvs_ip_o.setText("")
        self.folder_count = 1
        self.run_idx = 0

    def open_folder_picker(self):
        self.reset_file_selection()
        try:
            path = str(Path.home())
            input_folder = QFileDialog.getExistingDirectory(self, 'Select Input Folder', path)
            # print(input_folder)
            if len(os.listdir(input_folder)) > 0:
                for filename in os.listdir(input_folder):
                    full_file = os.path.join(input_folder, filename)
                    if filename.endswith(('.tif', '.png', '.jpeg', '.jpg')) and 'mask' not in filename.lower() and len(self.img_le.text()) == 0 or 'image' in filename.lower() and len(self.img_le.text()) == 0:
                        self.img_le.setText(full_file)
                    elif 'mask' in filename.lower() and filename.endswith(('.tif', '.tiff','.png', '.jpeg', '.jpg')) and 'image' not in filename.lower() and len(self.mask_le.text()) == 0:
                        self.mask_le.setText(full_file)
                    elif 'gold' in filename.lower() and filename.endswith('.csv') and 'landmark' not in filename.lower() and len(self.csv_le.text()) == 0:
                        self.csv_le.setText(full_file)
                    elif 'landmark' in filename.lower() and filename.endswith('.csv') and 'gold' not in filename.lower() and len(self.csv2_le.text()) == 0:
                        self.csv2_le.setText(full_file)
                    elif 'scalar' in filename.lower() and 'parameters' not in filename.lower() and filename.endswith('.txt') and len(self.scalar_le.text()) == 0:
                        self.set_scalar(full_file)
                        self.scalar_le.setText(full_file)
                    elif 'parameters' in filename.lower() and 'scalar' not in filename.lower() and filename.endswith('.txt') and len(self.param_le.text()) == 0:
                        self.set_params(full_file)
                        self.param_le.setText(full_file)
        except Exception as e:
            print(e, traceback.format_exc())

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
            elif len(self.param_le.text()) > 0:
                path = os.path.dirname(self.param_le.text())
            elif len(self.scalar_le.text()) > 0:
                path = os.path.dirname(self.scalar_le.text())    
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
                elif btn_type == FileType.PARAM: 
                    self.set_params(filename)
                    self.param_le.setText(filename)
                elif btn_type == FileType.SCALAR: 
                    self.set_scalar(filename)
                    self.scalar_le.setText(filename)
        except Exception as e:
            print(e, traceback.format_exc())

    def open_multi_folder_picker(self):
        self.multi_folders = []
        self.reset_file_selection()
        try:
            path = str(Path.home())
            input_folder_dir = QFileDialog.getExistingDirectory(self, 'Select Input Folder', path)
            detected_dirs = os.listdir(input_folder_dir)
            print('detected subdirectories:', detected_dirs)
            # remove hidden dirs
            removed_dirs = [f for f in detected_dirs if not os.path.isdir(os.path.join(input_folder_dir, f)) or f.startswith('.')] # may include parameters or scalar file
            detected_dirs = [f for f in detected_dirs if os.path.isdir(os.path.join(input_folder_dir, f)) and not f.startswith('.')]
            self.folder_count = len(detected_dirs)
            logging.info(f'folder count: {self.folder_count}')
            if (self.folder_count > 0):
                if (self.folder_count > 0) and os.path.isdir(input_folder_dir):
                    for subfolder_dir in detected_dirs:
                        full_sub_dir = os.path.join(input_folder_dir, subfolder_dir)
                        if (os.path.isdir(full_sub_dir)):
                            subfolder_contents = os.listdir(full_sub_dir)
                            if len(subfolder_contents) == 4: #> 0:
                                new_fld = {
                                    'image': [],
                                    'mask': [],
                                    'csv': [],
                                    'scalar': [],
                                    'parameters': []
                                }
                                for filename in subfolder_contents:
                                    full_file = os.path.join(full_sub_dir, filename)
                                    if filename.endswith(('.tif', '.png', '.jpeg', '.jpg')) and 'mask' not in filename.lower() and len(self.img_le.text()) == 0 or len(self.img_le.text()) == 0 and 'image' in filename.lower() and filename.endswith(('.tif', '.png', '.jpeg', '.jpg')):
                                        new_fld['image'].append(full_file)
                                    elif len(self.mask_le.text()) == 0 and filename.endswith(('.tif', '.tiff', '.png', '.jpeg', '.jpg')) and 'image' not in filename.lower() and 'mask' in filename.lower():
                                        new_fld['mask'].append(full_file)
                                    elif len(self.csv_le.text()) == 0 and filename.endswith('.csv') and 'landmark' not in filename.lower() and 'gold' in filename.lower():
                                        new_fld['csv'].append(full_file)
                                    elif len(self.csv2_le.text()) == 0 and filename.endswith('.csv') and 'gold' not in filename.lower() and 'landmark' in filename.lower():  
                                        new_fld['csv2'].append(full_file)
                                    elif filename.endswith('.txt') and 'scalar' in filename.lower()  and 'parameters' not in filename.lower() and len(self.scalar_le.text()) == 0: 
                                        new_fld['scalar'].append(full_file)
                                    elif 'parameters' in filename.lower() and 'scalar' not in filename.lower() and filename.endswith('.txt') and len(self.param_le.text()) == 0:
                                        new_fld['parameters'].append(full_file)
                                self.multi_folders.append(new_fld)
                            elif len(subfolder_contents) > 4:
                                new_fld = {
                                    'image': [],
                                    'mask': [],
                                    'csv': [],
                                    'csv2': [],
                                    'scalar': [],
                                    'parameters': []
                                }
                                for filename in subfolder_contents:
                                    full_file = os.path.join(full_sub_dir, filename)
                                    if filename.endswith(('.tif', '.png', '.jpeg', '.jpg')) and 'mask' not in filename.lower() and len(self.img_le.text()) == 0 or len(self.img_le.text()) == 0 and 'image' in filename.lower() and filename.endswith(('.tif', '.png', '.jpeg', '.jpg')):
                                        if 'image' not in filename.lower():
                                            if filename.endswith(('.tif', '.png', '.jpg')): 
                                                filename = filename.replace(filename[:-4], "image")
                                            elif filename.endswith(('.jpeg')):
                                                filename = filename.replace(filename[:-5], "image")
                                            new_fld['image'].append(full_file)
                                        else: 
                                            new_fld['image'].append(full_file)
                                    elif len(self.mask_le.text()) == 0 and filename.endswith(('.tif', '.tiff', '.png', '.jpeg', '.jpg')) and 'image' not in filename.lower() or len(self.mask_le.text()) == 0 and 'mask' in filename.lower() and filename.endswith(('.tif', '.png', '.jpeg', '.jpg')):
                                        new_fld['mask'].append(full_file)
                                    elif len(self.csv_le.text()) == 0 and 'landmark' not in filename.lower() and 'gold' in filename.lower() and len(self.csv_le.text()) == 0 and filename.endswith('.csv'):
                                        new_fld['csv'].append(full_file)
                                    elif len(self.csv2_le.text()) == 0 and 'gold' not in filename.lower() and 'landmark' in filename.lower() and len(self.csv2_le.text()) == 0 and filename.endswith('.csv'):  
                                        new_fld['csv2'].append(full_file)
                                    elif filename.endswith('.txt') and 'parameters' not in filename.lower() and 'scalar' in filename.lower() and len(self.scalar_le.text()) == 0: 
                                        new_fld['scalar'].append(full_file)
                                    elif filename.endswith('.txt') and 'scalar' not in filename.lower() or 'parameters' in filename.lower() and filename.endswith('.txt') and len(self.param_le.text()) == 0: 
                                        new_fld['parameters'].append(full_file)
                                self.multi_folders.append(new_fld)
                logging.info(f'folders: {self.multi_folders}')
                self.multi_folder_btn.setText(f'{self.folder_count} folders selected')
                if (len(self.multi_folders[0]['image']) > 0):
                    self.img_le.setText(self.multi_folders[0]['image'][0])
                if (len(self.multi_folders[0]['mask']) > 0):
                    self.mask_le.setText(self.multi_folders[0]['mask'][0])
                if (len(self.multi_folders[0]['csv']) > 0):
                    self.csv_le.setText(self.multi_folders[0]['csv'][0])
                #if ['csv2'] in self.multi_folders[0]:
                #    if (len(self.multi_folders[0]['csv2']) > 0):
                #        self.csv2_le.setText(self.multi_folders[0]['csv2'][0])
                #    self.csv_le.setText(self.multi_folders[0]['csv'][0])
                #     self.csv2_le.setText('None Selected (CSV)')
                if (len(self.multi_folders[0]) == 5):
                    print(self.multi_folders[0])
                    logging.info("Landmark file not detected. Gold Rippler and Gold Star work optimally with a landmark.")
                elif (len(self.multi_folders[0]['csv2']) > 0):
                    self.csv2_le.setText(self.multi_folders[0]['csv2'][0])
                if (len(self.multi_folders[0]['scalar']) > 0):
                    self.scalar_le.setText(self.multi_folders[0]['scalar'][0])
                    self.set_scalar(self.multi_folders[0]['scalar'][0])
                elif (len(self.multi_folders[0]['scalar']) == 0):
                    for subfolder_dir in removed_dirs:
                        if subfolder_dir.endswith('.txt') and 'parameters' not in subfolder_dir.lower() and 'scalar' in subfolder_dir.lower(): 
                            subfolder_dir = os.path.join(input_folder_dir, subfolder_dir)
                            self.scalar_le.setText(subfolder_dir)
                            self.set_scalar(subfolder_dir)
                if (len(self.multi_folders[0]['parameters']) > 0):
                    self.param_le.setText(self.multi_folders[0]['parameters'][0])
                    self.set_params(self.multi_folders[0]['parameters'][0])
                elif (len(self.multi_folders[0]['parameters']) == 0):
                    for subfolder_dir in removed_dirs:
                        if 'parameters' in subfolder_dir.lower() and subfolder_dir.endswith('.txt') and len(self.param_le.text()) == 0: 
                            subfolder_dir = os.path.join(input_folder_dir, subfolder_dir)
                            self.param_le.setText(subfolder_dir)
                            self.set_params(subfolder_dir)
        except Exception as e:
            print(e, traceback.format_exc())

    def open_output_folder_picker(self):
        self.output_dir_le.setText(QFileDialog.getExistingDirectory(self, 'Select Output Folder'))

    def set_scalar(self, scalar_file):
        try:
            with open(scalar_file, 'r') as f:
                # read in scalar file and extract value and unit
                lines = f.read()
                scalars = lines.split(' ')
                if len(scalars) > 1:
                    scalar_in = scalars[0][(int(scalars[0].lower().find("1px=",0))+4):].lower()
                    scalar_out = scalars[1][(int(scalars[1].lower().find("1px=",0))+4):].lower()
                elif len(scalars) == 1: # converting between px and another unit 
                    scalar_in = scalars[0][:(int(scalars[0].lower().find("1px=",0))+3)].lower()
                    scalar_out = scalars[0][(int(scalars[0].lower().find("1px=",0))+4):].lower()
                # set input scalar accordingly
                if 'px' in scalar_in:
                    scalar_in = scalar_in.replace('px', '')
                    self.ip_scalar_type.setCurrentIndex(0)
                elif 'nm' in scalar_in:
                    scalar_in = scalar_in.replace('nm', '')
                    self.ip_scalar_type.setCurrentIndex(1)
                elif 'um' in scalar_in or 'µm' in scalar_in:
                    scalar_in = scalar_in.replace('um', '').replace('μm', '')
                    self.ip_scalar_type.setCurrentIndex(2)
                self.csvs_ip_i.setText(scalar_in)
                # set output scalar accordingly
                if 'px' in scalar_out:
                    scalar_out = scalar_out.replace('px', '')
                    self.op_scalar_type.setCurrentIndex(0)
                elif 'nm' in scalar_out:
                    scalar_out = scalar_out.replace('nm', '')
                    self.op_scalar_type.setCurrentIndex(1)
                elif 'um' in scalar_out or 'µm' in scalar_out:
                   #scalar_out = scalar_out.replace('um', '').replace('μm', '')
                   scalar_out = scalar_out[:-3]
                   self.op_scalar_type.setCurrentIndex(2)
                self.csvs_ip_o.setText(scalar_out)
        except Exception as e:
            print(e, traceback.format_exc())
    
    def set_params(self, parameters_file):
        try:
            with open(parameters_file, 'r') as f:
                # Format: 
                # Distance threshold=27px (requires a space after each line)
                # Min clust size=2px
                # Random trials=1
                # Maximum steps=10
                # Step size=60px 
                # Initial radius=50px 
                lines = f.read()
                parameter_input = lines.split(' ') # denotes lines of the .txt \n
                first_line = parameter_input[0]
                distance_thresh = first_line[-4:]
                if '=' in distance_thresh: # if it is less than 10 
                    distance_thresh = first_line[-3:]
                if 'px' in distance_thresh: 
                    distance_thresh = distance_thresh.replace('px', '')
                distance_thresh = int(distance_thresh)
                self.parameters[0] = distance_thresh
                if len(parameter_input) > 1 and parameter_input[1] != '':
                    second_line = parameter_input[1]
                    clust_size = second_line[7:]
                    clust_size = int(clust_size) 
                    self.parameters[1] = clust_size
                if len(parameter_input) > 2 and parameter_input[2] != '': # a variable for random exists 
                    random_trials = parameter_input[2]
                    random_coords.N = int(random_trials[8:])
                self.params_row_1.repaint()
                self.params_row_1.setText(f'Distance threshold: {self.parameters[0]}px           Minimum cluster size: {self.parameters[1]}           # random trials: {random_coords.N}           ')
                if len(parameter_input) > 3 and parameter_input[3] != '': # if there are more than 2 or 3 lines 
                    third_line = parameter_input[3]
                    no_steps = third_line[7:]
                    no_steps = int(no_steps)
                    self.parameters[2] = no_steps
                if len(parameter_input) > 4 and parameter_input[3] != '':
                    fourth_line = parameter_input[4]
                    step_size = fourth_line[6:]
                    if 'px' in step_size: 
                        step_size = step_size.replace('px', '')
                    step_size = int(step_size)
                    self.parameters[3] = step_size
                if len(parameter_input) > 5 and parameter_input[3] != '':
                    fifth_line = parameter_input[5]
                    radius = fifth_line[8:]
                    if 'px' in radius: 
                        radius = radius.replace('px', '')
                    radius = int(radius)
                    self.parameters[4] = radius
                    self.params_row_2.repaint()
                    self.params_row_2.setText(f'Maximum # steps: {self.parameters[2]}               Step size: {self.parameters[3]}px                         Initial radius: {self.parameters[4]}px')
        except Exception as e:
            print(e, traceback.format_exc()) 

    def reload_params_func(self):
        self.param_le.setText("")
        distance_thresh = 27
        self.parameters[0] = distance_thresh
        clust_size = 2
        self.parameters[1] = clust_size
        random_trials = 1
        random_coords.N = random_trials
        self.params_row_1.repaint()
        self.params_row_1.setText(f'Distance threshold: {self.parameters[0]}px           Minimum cluster size: {self.parameters[1]}           # random trials: {random_coords.N}           ')
        no_steps = 10
        self.parameters[2] = no_steps
        step_size = 60
        self.parameters[3] = step_size
        radius = 50
        self.parameters[4] = radius
        self.params_row_2.repaint()
        self.params_row_2.setText(f'Maximum # steps: {self.parameters[2]}               Step size: {self.parameters[3]}px                         Initial radius: {self.parameters[4]}px')    


    def set_mask_clr(self):
        """ MASK COLOR SET """
        color = QColorDialog.getColor().name(0)
        # print(color)
        comp_color = get_complimentary_color(color)
        self.clr_btn.setStyleSheet(
            f'QWidget {{background-color: {color}; font-size: 16px; font-weight: 600; padding: 8px; color: {comp_color}; border-radius: 7px; }}')
        self.clr_btn.setText(color)
