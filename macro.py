# QT5
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QTextEdit, QAction, QFileDialog, QApplication,
                             QSpacerItem, QDialog, QRadioButton, QCheckBox, QHBoxLayout, QGraphicsColorizeEffect,
                             QPushButton, QWidget, QGridLayout, QSizePolicy, QFormLayout, QLineEdit, QColorDialog,
                             QComboBox, QProgressBar)
from functools import partial
import cv2

from image_viewer import QImageViewer
from nnd import run_nnd, draw_length
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from utils import Progress, create_color_pal


class MacroPage(QWidget):
    def __init__(self, header_name="Undefined", desc="Undefined", img_dropdown=None, mask_dropdown=None, csv_dropdown=None,
                 parameters=None):
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
        # knn specific props
        self.csvs_lb = QLabel("CSV Scalar")
        self.csvs_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.csvs_ip = QLineEdit()
        self.csvs_ip.setStyleSheet("font-size: 16px; padding: 8px;  font-weight: 400; background: #ddd; border-radius: 7px;  margin-bottom: 5px; max-width: 75px;")
        self.csvs_ip.setPlaceholderText("1")
        self.gen_rand_lb = QLabel("Generate Rand Coords")
        self.gen_rand_lb.setStyleSheet("margin-left: 50px; font-size: 17px; font-weight: 400;")
        self.gen_rand_cb = QCheckBox()
        self.nnd_props = QHBoxLayout()
        self.nnd_props.addWidget(self.csvs_lb)
        self.nnd_props.addWidget(self.csvs_ip)
        self.nnd_props.addWidget(self.gen_rand_lb)
        self.nnd_props.addWidget(self.gen_rand_cb)
        layout.addRow(self.nnd_props)
        # hist specific props
        self.bars_lb = QLabel("# Bars in Histogram")
        self.bars_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.bars_ip = QLineEdit()
        self.bars_ip.setStyleSheet("font-size: 16px; padding: 8px;  font-weight: 400; background: #ddd; border-radius: 7px;  margin-bottom: 5px; max-width: 75px;")
        self.bars_ip.setPlaceholderText("10")
        self.hist_props = QHBoxLayout()
        self.hist_props.addWidget(self.bars_lb)
        self.hist_props.addWidget(self.bars_ip)
        layout.addRow(self.hist_props)

        # output header
        self.out_header = QLabel("Output")
        layout.addRow(self.out_header)

        self.out_desc = QLabel("Double-click on an image to open it.")
        self.out_desc.setStyleSheet("font-size: 17px; font-weight: 400; padding-top: 3px; padding-bottom: 20px;")
        self.out_desc.setWordWrap(True)
        layout.addRow(self.out_desc)

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
        self.run_btn.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px; margin-top: 3px; background: #ff8943; color: white; border-radius: 7px; ")
        self.run_btn.clicked.connect(self.run)
        self.download_btn = QPushButton('Download', self)
        self.download_btn.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px; margin-top: 3px; background: #ccc; color: white; border-radius: 7px; ")
        self.download_btn.clicked.connect(partial(self.download, "nnd_output.csv"))
        btn_r = QHBoxLayout()
        btn_r.addWidget(self.run_btn)
        btn_r.addWidget(self.download_btn)
        layout.addRow(btn_r)

        # assign layout
        self.setLayout(layout)

        # run on init
        self.run()

    def on_progress_update(self, value):
        self.progress.setValue(value)

    def run(self):
        try:
            prog_wrapper = Progress()
            prog_wrapper.prog.connect(self.on_progress_update)
            self.palette = create_color_pal(n_bins=self.bars_ip.text() if len(self.bars_ip.text()) > 0 else 10)

            # run knn
            self.OUTPUT_DF = run_nnd(prog_wrapper=prog_wrapper, img_path=self.img_drop.currentText(), csv_path=self.csv_drop.currentText(),
                                         pface_path="",
                                         csv_scalar=(self.csvs_ip.text() if len(self.csvs_ip.text()) > 0 else 1),
                                         gen_rand=self.gen_rand_cb.isChecked())
            self.progress.setValue(100)
            # get drawn img
            drawn_img = draw_length(nnd_df=self.OUTPUT_DF, palette=self.palette, img=cv2.imread(self.img_drop.currentText()))
            self.show_image(drawn_img)
            self.create_hist(self.bars_ip.text() if self.bars_ip.text() else 10)

            print(self.OUTPUT_DF.head())
            self.download_btn.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px; margin-top: 3px; background: teal; color: white; border-radius: 7px; ")
        except Exception as e:
            print(e)

    def download(self, file_name):
        if self.OUTPUT_DF.shape[0] > 0 and self.OUTPUT_DF.shape[1] > 0:
            try:
                self.OUTPUT_DF.to_csv(file_name, index=False, header=True)
            except Exception as e:
                print(e)

    def show_image(self, img):
        self.display_img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(self.display_img)
        smaller_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.image_frame.setPixmap(smaller_pixmap)

    def create_hist(self, n_bins=10, cmap='RdYlBu_r'):
        cm = plt.cm.get_cmap(cmap)
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # create hist
        n, bins, patches = ax.hist(self.OUTPUT_DF['dist'], bins=n_bins, color='green')
        ax.set_xlabel("N Nearest Distance (nm)")
        ax.set_ylabel("Number of Entries")
        ax.set_title('Distances Between Nearest Neighbors')

        # normalize values
        col = (n - n.min()) / (n.max() - n.min())
        # colormap
        for c, p in zip(col, patches):
            p.set_facecolor(cm(c))
        canvas.draw()
        size = canvas.size()
        width, height = size.width(), size.height()
        self.hist = QImage(canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
        # display img
        pixmap = QPixmap.fromImage(self.hist)
        smaller_pixmap = pixmap.scaled(400, 250, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.hist_frame.setPixmap(smaller_pixmap)

    def open_large(self, event, file):
        self.image_viewer = QImageViewer(file)
        self.image_viewer.show()
