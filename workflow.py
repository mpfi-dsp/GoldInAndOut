# pyQT5
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QLabel, QRadioButton, QCheckBox, QHBoxLayout, QPushButton, QWidget, QSizePolicy,
                             QFormLayout, QLineEdit,
                             QComboBox, QProgressBar)
# general
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from image_viewer import QImageViewer
from functools import partial
import seaborn as sns
import cv2
# utils
from globals import PALETTE_OPS
from utils import Progress, create_color_pal, download_csv
from nnd import run_nnd, draw_length

""" 
WORKFLOW PAGE
__________________
@workflow: selected workflow

"""


class WorkflowPage(QWidget):
    def __init__(self, scaled_df, workflow=None, csv_scalar=1, header_name="Undefined", desc="Undefined", img_dropdown=None,
                 mask_dropdown=None, csv_dropdown=None, input_unit='px', scalar=1,
                 props=None):
        super().__init__()
        if img_dropdown is None:
            img_dropdown = []
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

        """ PARAMETERS """
        self.workflows_header = QLabel("Parameters")
        layout.addRow(self.workflows_header)
        self.csv_lb = QLabel("CSV")
        self.csv_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.csv_drop = QComboBox()
        self.csv_drop.addItems(csv_dropdown)
        layout.addRow(self.csv_lb, self.csv_drop)
        # hist specific props
        self.bars_lb = QLabel("# Bins in Histogram")
        self.bars_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.bars_ip = QLineEdit()
        self.bars_ip.setStyleSheet(
            "font-size: 16px; padding: 8px;  font-weight: 400; background: #ddd; border-radius: 7px;  margin-bottom: 5px; max-width: 75px;")
        self.bars_ip.setPlaceholderText("10")
        self.pal_lb = QLabel("Histogram Color Palette")
        self.pal_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.pal_type = QComboBox()
        self.pal_type.addItems(PALETTE_OPS)
        layout.addRow(self.pal_lb, self.pal_type)
        layout.addRow(self.bars_lb, self.bars_ip)

        """ ADVANCED: RANDOM COORDS SECTION """
        self.gen_rand_head = QLabel("Advanced: Generate Rand Coords")
        self.gen_rand_head.setStyleSheet("font-size: 17px; font-weight: 500;")
        self.gen_rand_adv_cb = QRadioButton()
        self.gen_rand_adv_cb.setStyleSheet("font-size: 17px; font-weight: 500; padding-top: 6px;")
        self.gen_rand_adv_cb.clicked.connect(self.toggle_adv)
        layout.addRow(self.gen_rand_head, self.gen_rand_adv_cb)

        self.img_lb = QLabel("Image")
        self.img_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.img_drop = QComboBox()
        self.img_drop.addItems(img_dropdown)
        layout.addRow(self.img_lb, self.img_drop)
        self.mask_lb = QLabel("Mask")
        self.mask_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.mask_drop = QComboBox()
        self.mask_drop.addItems(mask_dropdown)
        layout.addRow(self.mask_lb, self.mask_drop)

        self.n_coord_lb = QLabel("# Random Coordinates To Generate")
        self.n_coord_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.n_coord_ip = QLineEdit()
        self.n_coord_ip.setStyleSheet(
            "font-size: 16px; padding: 8px;  font-weight: 400; background: #ddd; border-radius: 7px;  margin-bottom: 5px; max-width: 200px;")
        self.n_coord_ip.setPlaceholderText("default is # in real csv")
        layout.addRow(self.n_coord_lb, self.n_coord_ip)
        self.n_coord_lb.setHidden(True)
        self.n_coord_ip.setHidden(True)
        self.img_drop.setHidden(True)
        self.img_lb.setHidden(True)
        self.mask_lb.setHidden(True)
        self.mask_drop.setHidden(True)

        # output header
        self.out_header = QLabel("Output")
        layout.addRow(self.out_header)

        self.out_desc = QLabel("Double-click on an image to open it.")
        self.out_desc.setStyleSheet("font-size: 17px; font-weight: 400; padding-top: 3px; padding-bottom: 20px;")
        self.out_desc.setWordWrap(True)
        layout.addRow(self.out_desc)

        self.gen_real_lb = QLabel("Display Real Coords")
        self.gen_real_lb.setStyleSheet("margin-left: 50px; font-size: 17px; font-weight: 400;")
        self.gen_real_cb = QCheckBox()
        self.gen_real_cb.setChecked(True)

        self.gen_rand_lb = QLabel("Display Random Coords")
        self.gen_rand_lb.setStyleSheet("margin-left: 50px; font-size: 17px; font-weight: 400;")
        self.gen_rand_cb = QCheckBox()

        cb_row = QHBoxLayout()
        cb_row.addWidget(self.gen_real_lb)
        cb_row.addWidget(self.gen_real_cb)
        cb_row.addWidget(self.gen_rand_lb)
        cb_row.addWidget(self.gen_rand_cb)
        layout.addRow(cb_row)

        # annotated image
        self.image_frame = QLabel()
        self.image_frame.setStyleSheet("padding-top: 3px; background: white;")
        self.image_frame.setMaximumSize(400, 250)
        self.image_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_frame.mouseDoubleClickEvent = lambda event: self.open_large(event, self.display_img)

        # hist
        self.hist_frame = QLabel()
        self.hist_frame.setStyleSheet("padding-top: 3px; background: white;")
        self.hist_frame.setMaximumSize(400, 250)
        self.hist_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.hist_frame.mouseDoubleClickEvent = lambda event: self.open_large(event, self.hist)

        # container for visualizers
        self.img_cont = QHBoxLayout()
        self.img_cont.addWidget(self.image_frame)
        self.img_cont.addWidget(self.hist_frame)
        layout.addRow(self.img_cont)

        self.progress = QProgressBar(self)
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMaximum(100)
        layout.addRow(self.progress)

        # run & download btns
        self.run_btn = QPushButton('Run Again', self)
        self.run_btn.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 8px; margin-top: 3px; background: #E89C12; color: white; border-radius: 7px; ")
        self.run_btn.clicked.connect(partial(self.run, scaled_df, scalar, input_unit))
        self.download_btn = QPushButton('Download', self)
        self.download_btn.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 8px; margin-top: 3px; background: #ccc; color: white; border-radius: 7px; ")
        self.download_btn.clicked.connect(partial(download_csv, self.OUTPUT_DF, "nnd_output.csv"))
        btn_r = QHBoxLayout()
        btn_r.addWidget(self.run_btn)
        btn_r.addWidget(self.download_btn)
        layout.addRow(btn_r)

        # assign layout
        self.setLayout(layout)

        # run on init
        self.run(scaled_df=scaled_df, scalar=scalar, input_unit=input_unit)

    def on_progress_update(self, value):
        self.progress.setValue(value)

    def toggle_adv(self):
        self.img_lb.setVisible(not self.img_lb.isVisible())
        self.img_drop.setVisible(not self.img_drop.isVisible())
        self.mask_lb.setVisible(not self.mask_lb.isVisible())
        self.mask_drop.setVisible(not self.mask_drop.isVisible())
        self.n_coord_ip.setVisible(not self.n_coord_ip.isVisible())
        self.n_coord_lb.setVisible(not self.n_coord_lb.isVisible())

    def run(self, scaled_df, scalar, input_unit):
        try:
            prog_wrapper = Progress()
            prog_wrapper.prog.connect(self.on_progress_update)

            # run knn
            self.REAL_COORDS, self.RAND_COORDS = run_nnd(
                                    data=scaled_df,
                                    prog_wrapper=prog_wrapper,
                                     img_path=self.img_drop.currentText(),
                                     # csv_path=self.csv_drop.currentText(),
                                     pface_path=self.mask_drop.currentText(),
                                    )
            # print("real", self.REAL_COORDS, "rand", self.RAND_COORDS)
            self.progress.setValue(100)
            print(self.RAND_COORDS.head())

            self.create_visuals(n_bins=(self.bars_ip.text() if self.bars_ip.text() else 'fd'), scalar=scalar, input_unit=input_unit)
            self.download_btn.setStyleSheet(
                "font-size: 16px; font-weight: 600; padding: 8px; margin-top: 3px; background: #007267; color: white; border-radius: 7px; ")
        except Exception as e:
            print(e)

    def create_visuals(self, n_bins='fd', input_unit='px', scalar=1):
        cm = sns.color_palette(self.pal_type.currentText(), as_cmap=True)
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        self.REAL_COORDS.sort_values('dist', inplace=True)
        if self.gen_rand_cb.isChecked():
            self.RAND_COORDS.sort_values('dist', inplace=True)

        # create hist
        n, bins, patches = ax.hist(self.REAL_COORDS['dist'], bins=n_bins, color='green')
        ax.set_xlabel(f'Nearest Neighbor Distance ({input_unit})')
        ax.set_ylabel("Number of Entries")
        ax.set_title('Distances Between Nearest Neighbors')

        # generate palette
        palette = create_color_pal(n_bins=int(len(n)), palette_type=self.pal_type.currentText())

        # normalize values
        col = (n - n.min()) / (n.max() - n.min())
        for c, p in zip(col, patches):
            p.set_facecolor(cm(c))

        canvas.draw()
        size = canvas.size()
        width, height = size.width(), size.height()
        self.hist = QImage(canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
        # display img
        pixmap = QPixmap.fromImage(self.hist)
        smaller_pixmap = pixmap.scaled(300, 250, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.hist_frame.setPixmap(smaller_pixmap)

        drawn_img = cv2.imread(self.img_drop.currentText())
        # real coords
        if self.gen_real_cb.isChecked():
            drawn_img = draw_length(nnd_df=self.REAL_COORDS, bin_counts=n, palette=palette, input_unit=input_unit,
                                    scalar=scalar, img=drawn_img)

        # random
        if self.gen_rand_cb.isChecked():
            drawn_img = draw_length(nnd_df=self.RAND_COORDS, bin_counts=n, palette=palette, input_unit=input_unit,
                                    scalar=scalar, img=drawn_img)

        self.display_img = QImage(drawn_img.data, drawn_img.shape[1], drawn_img.shape[0],
                                  QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(self.display_img)
        smaller_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.image_frame.setPixmap(smaller_pixmap)

    def open_large(self, event, file):
        self.image_viewer = QImageViewer(file)
        self.image_viewer.show()
