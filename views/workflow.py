# pyQT5
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtWidgets import (QLabel, QRadioButton, QCheckBox, QHBoxLayout, QPushButton, QWidget, QSizePolicy,
                             QFormLayout, QLineEdit,
                             QComboBox, QProgressBar, QToolButton)
# general
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from views.image_viewer import QImageViewer
from functools import partial
import seaborn as sns
import cv2
# utils
from globals import PALETTE_OPS
from typings import Unit, Workflow
from utils import Progress, create_color_pal, download_csv, pixels_conversion_w_distance, enum_to_unit
from workflows.clust import run_clust, draw_clust
from workflows.nnd import run_nnd, draw_length
from workflows.random import gen_random_coordinates

""" 
WORKFLOW PAGE
__________________
@scaled_df: dataframe containing csv coordinate data of gold particles scaled via scalar to proper unit
@workflow: JSON object containing the following data:
    @type: ENUM type of Workflow
    @header: string displayed as "header"
    @desc: string displayed as "description" below header
    @hist: histogram metadata:
        @title: title of histogram
        @x_label: x_label of histogram
        @y_label: y_label of histogram
@img: array of selected image paths
@mask: array of selected mask paths
@csv: array of selected csv paths
@input_unit: metric input unit
@output_unit: metric output unit
@scalar: multiplier ratio between input metric unit (usually pixels) and desired output metric unit
"""


class WorkflowPage(QWidget):
    def __init__(self, scaled_df, workflow=None, img=None, mask=None, csv=None, input_unit=Unit.PIXEL,
                 output_unit=Unit.PIXEL, scalar=1):
        super().__init__()

        self.OUTPUT_DF = pd.DataFrame()
        layout = QFormLayout()
        # header
        header = QLabel(workflow['header'])
        header.setStyleSheet("font-size: 24px; font-weight: bold; padding-top: 8px; ")
        layout.addRow(header)
        desc = QLabel(workflow['desc'])
        desc.setStyleSheet("font-size: 17px; font-weight: 400; padding-top: 3px; padding-bottom: 20px;")
        desc.setWordWrap(True)
        layout.addRow(desc)

        """ REUSABLE PARAMETERS """
        self.workflows_header = QLabel("Parameters")
        layout.addRow(self.workflows_header)

        if len(workflow['props']) > 0:
            self.p1_l = QLabel(workflow['props'][0]['title'])
            self.p1_l.setStyleSheet("font-size: 17px; font-weight: 400;")
            self.p1_ip = QLineEdit()
            self.p1_ip.setPlaceholderText(workflow['props'][0]['placeholder'])
            layout.addRow(self.p1_l, self.p1_ip)

        if len(workflow['props']) > 1:
            self.p2_l = QLabel(workflow['props'][1]['title'])
            self.p2_l.setStyleSheet("font-size: 17px; font-weight: 400;")
            self.p2_ip = QLineEdit()
            self.p2_ip.setPlaceholderText(workflow['props'][1]['placeholder'])
            layout.addRow(self.p2_l, self.p2_ip)

        if len(workflow['props']) > 2:
            self.p3_l = QLabel(workflow['props'][2]['title'])
            self.p3_l.setStyleSheet("font-size: 17px; font-weight: 400;")
            self.p3_ip = QLineEdit()
            self.p3_ip.setPlaceholderText(workflow['props'][2]['placeholder'])
            layout.addRow(self.p3_l, self.p3_ip)

        """ REAL COORDS SECTION """
        self.gen_head = QLabel("Real Coordinates")
        self.gen_head.setStyleSheet("font-size: 17px; font-weight: 500; padding-top: 0px; padding-bottom: 0px; margin-top: 0px; margin-bottom: 0px;")
        self.gen_head_cb = QToolButton()
        self.gen_head_cb.setArrowType(Qt.DownArrow)
        self.gen_head_cb.setCursor(QCursor(Qt.PointingHandCursor))
        self.gen_head_cb.clicked.connect(self.toggle_gen_adv)
        layout.addRow(self.gen_head, self.gen_head_cb)
        # csv
        self.csv_lb = QLabel("csv")
        self.csv_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.csv_drop = QComboBox()
        self.csv_drop.addItems(csv)
        layout.addRow(self.csv_lb, self.csv_drop)
        # num bins
        self.bars_lb = QLabel(
            '<a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html"># hist bins</a>')
        self.bars_lb.setOpenExternalLinks(True)
        self.bars_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.bars_ip = QLineEdit()
        self.bars_ip.setPlaceholderText("10 OR [1, 2, 3, 4] OR 'fd'")
        layout.addRow(self.bars_lb, self.bars_ip)
        # color palette
        self.pal_lb = QLabel('<a href="https://seaborn.pydata.org/tutorial/color_palettes.html">color palette</a>')
        self.pal_lb.setOpenExternalLinks(True)
        self.pal_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.pal_type = QComboBox()
        self.pal_type.addItems(PALETTE_OPS)
        layout.addRow(self.pal_lb, self.pal_type)
        for prop in [self.csv_lb, self.csv_drop, self.pal_lb, self.pal_type, self.bars_lb, self.bars_ip]:
            prop.setHidden(True)

        """ RANDOM COORDS SECTION """
        self.gen_rand_head = QLabel("Random Coordinates")
        self.gen_rand_head.setStyleSheet("font-size: 17px; font-weight: 500; padding-top: 0px; padding-bottom: 0px; margin-top: 0px; margin-bottom: 0px;")
        self.gen_rand_adv_cb = QToolButton()
        self.gen_rand_adv_cb.setArrowType(Qt.DownArrow)
        self.gen_rand_adv_cb.setCursor(QCursor(Qt.PointingHandCursor))
        self.gen_rand_adv_cb.clicked.connect(self.toggle_rand_adv)
        layout.addRow(self.gen_rand_head, self.gen_rand_adv_cb)
        # image path
        self.img_lb = QLabel("image")
        self.img_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.img_drop = QComboBox()
        self.img_drop.addItems(img)
        layout.addRow(self.img_lb, self.img_drop)  # csv
        # mask path
        self.mask_lb = QLabel("mask")
        self.mask_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.mask_drop = QComboBox()
        self.mask_drop.addItems(mask)
        layout.addRow(self.mask_lb, self.mask_drop)
        # palette random
        self.r_pal_lb = QLabel(
            '<a href="https://seaborn.pydata.org/tutorial/color_palettes.html">rand color palette</a>')
        self.r_pal_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.r_pal_lb.setOpenExternalLinks(True)
        self.r_pal_type = QComboBox()
        self.r_pal_type.addItems(PALETTE_OPS)
        self.r_pal_type.setCurrentText('crest')
        layout.addRow(self.r_pal_lb, self.r_pal_type)
        # num coords to gen
        self.n_coord_lb = QLabel("# of coords")
        self.n_coord_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.n_coord_ip = QLineEdit()
        self.n_coord_ip.setStyleSheet(
            "font-size: 16px; padding: 8px;  font-weight: 400; background: #ddd; border-radius: 7px;  margin-bottom: 5px; ") # max-width: 200px;
        self.n_coord_ip.setPlaceholderText("default is # in real csv")
        layout.addRow(self.n_coord_lb, self.n_coord_ip)
        # set adv hidden by default
        for prop in [self.img_lb, self.img_drop, self.mask_lb, self.mask_drop, self.n_coord_lb, self.n_coord_ip,
                     self.r_pal_type, self.r_pal_lb]:
            prop.setHidden(True)
        # output header
        self.out_header = QLabel("Output")
        layout.addRow(self.out_header)
        # toggleable output
        self.out_desc = QLabel("Check boxes to toggle output options. Double-click on an image to open it.")
        self.out_desc.setStyleSheet("font-size: 17px; font-weight: 400; padding-top: 3px; padding-bottom: 20px;")
        self.out_desc.setWordWrap(True)
        layout.addRow(self.out_desc)
        # real
        self.gen_real_lb = QLabel("display real coords")
        self.gen_real_lb.setStyleSheet("margin-left: 50px; font-size: 17px; font-weight: 400;")
        self.gen_real_cb = QCheckBox()
        self.gen_real_cb.setChecked(True)
        # rand
        self.gen_rand_lb = QLabel("display random coords")
        self.gen_rand_lb.setStyleSheet("margin-left: 50px; font-size: 17px; font-weight: 400;")
        self.gen_rand_cb = QCheckBox()
        # cb row
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
        self.image_frame.setCursor(QCursor(Qt.PointingHandCursor))
        self.image_frame.mouseDoubleClickEvent = lambda event: self.open_large(event, self.display_img)
        # hist
        self.hist_frame = QLabel()
        self.hist_frame.setStyleSheet("padding-top: 3px; background: white;")
        self.hist_frame.setMaximumSize(400, 250)
        self.hist_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.hist_frame.mouseDoubleClickEvent = lambda event: self.open_large(event, self.hist)
        self.hist_frame.setCursor(QCursor(Qt.PointingHandCursor))
        # container for visualizers
        self.img_cont = QHBoxLayout()
        self.img_cont.addWidget(self.image_frame)
        self.img_cont.addWidget(self.hist_frame)
        layout.addRow(self.img_cont)
        # loading bar
        self.progress = QProgressBar(self)
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMaximum(100)
        layout.addRow(self.progress)
        # run & download btns
        self.run_btn = QPushButton('Run Again', self)
        self.run_btn.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 8px; margin-top: 3px; background: #E89C12; color: white; border-radius: 7px; ")
        self.run_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.run_btn.clicked.connect(partial(self.run, workflow, scaled_df, scalar, input_unit, output_unit))
        self.download_btn = QPushButton('Download', self)
        self.download_btn.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 8px; margin-top: 3px; background: #ccc; color: white; border-radius: 7px; ")
        self.download_btn.clicked.connect(partial(self.download, output_unit, workflow))
        self.download_btn.setCursor(QCursor(Qt.PointingHandCursor))
        btn_r = QHBoxLayout()
        btn_r.addWidget(self.run_btn)
        btn_r.addWidget(self.download_btn)
        layout.addRow(btn_r)
        # assign layout
        self.setLayout(layout)
        # run on init
        self.run(workflow, scaled_df, scalar, input_unit, output_unit)

    """ UPDATE PROGRESS BAR """
    def on_progress_update(self, value):
        self.progress.setValue(value)

    """ DOWNLOAD FILES """
    def download(self, output_unit, workflow):
        self.download_btn.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 8px; margin-top: 3px; background: #EBBA22; color: white; border-radius: 7px; ")
        self.download_btn.setText("Download Again")
        try:
            if self.gen_real_cb.isChecked():
                download_csv(self.real_df,
                             f'{workflow["name"].lower()}/real_{workflow["name"].lower()}_output_{enum_to_unit(output_unit)}.csv')
            if self.gen_rand_cb.isChecked():
                download_csv(self.rand_df,
                             f'{workflow["name"].lower()}/rand_{workflow["name"].lower()}_output_{enum_to_unit(output_unit)}.csv')
            self.display_img.save(f'./output/{workflow["name"].lower()}/drawn_{workflow["name"].lower()}_img.tif')
            self.hist.save(f'./output/{workflow["name"].lower()}/{workflow["name"].lower()}_histogram.jpg')
        except Exception as e:
            print(e)

    """ TOGGLE GENERAL ADV OPTIONS """

    def toggle_gen_adv(self):
        print(self.gen_head_cb.arrowType())
        self.gen_head_cb.setArrowType(Qt.UpArrow if self.gen_head_cb.arrowType() == Qt.DownArrow else Qt.DownArrow)
        for prop in [self.csv_lb, self.csv_drop, self.pal_lb, self.pal_type, self.bars_lb, self.bars_ip]:
            prop.setVisible(not prop.isVisible())

    """ TOGGLE RAND ADV OPTIONS """
    def toggle_rand_adv(self):
        self.gen_rand_adv_cb.setArrowType(Qt.UpArrow if self.gen_rand_adv_cb.arrowType() == Qt.DownArrow else Qt.DownArrow)
        for prop in [self.img_lb, self.img_drop, self.mask_lb, self.mask_drop, self.n_coord_lb, self.n_coord_ip,
                     self.r_pal_type, self.r_pal_lb]:
            prop.setVisible(not prop.isVisible())

    """ RUN WORKFLOW """
    def run(self, workflow, scaled_df, scalar, input_unit, output_unit):
        try:
            prog_wrapper = Progress()
            prog_wrapper.prog.connect(self.on_progress_update)

            # generate random coords
            random_coords = gen_random_coordinates(
                data=scaled_df, img_path=self.img_drop.currentText(),
                                                   pface_path=self.mask_drop.currentText(),
                                                   n_rand_to_gen=int(self.n_coord_ip.text()) if self.n_coord_ip.text() else len(scaled_df.index))

            # select workflow
            if workflow["type"] == Workflow.NND:
                self.real_df, self.rand_df = run_nnd(df=scaled_df, prog=prog_wrapper, random_coordinate_list=random_coords)
            elif workflow["type"] == Workflow.CLUST:
                self.real_df, self.rand_df = run_clust(df=scaled_df, random_coordinate_list=random_coords, prog=prog_wrapper, distance_threshold=120)

            self.progress.setValue(100)
            # create ui scheme
            self.create_visuals(workflow=workflow, n_bins=(self.bars_ip.text() if self.bars_ip.text() else 'fd'),
                                input_unit=input_unit, output_unit=output_unit, scalar=scalar)
            self.download_btn.setStyleSheet(
                "font-size: 16px; font-weight: 600; padding: 8px; margin-top: 3px; background: #007267; color: white; border-radius: 7px; ")
        except Exception as e:
            print(e)

    """ CREATE DATA VISUALIZATIONS """
    def create_visuals(self, workflow, n_bins, input_unit, output_unit, scalar):
        # init vars & figure
        hist_df = pd.DataFrame([])
        cm = plt.cm.get_cmap('crest')
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # create hist
        if self.gen_real_cb.isChecked():
            self.real_df.sort_values(workflow["hist"]["x_type"], inplace=True)
            hist_df = self.real_df[workflow["hist"]["x_type"]]
            cm = sns.color_palette(self.pal_type.currentText(), as_cmap=True)
            ax.set_title(f'{workflow["hist"]["title"]} (Real)')
        elif self.gen_rand_cb.isChecked():
            self.rand_df.sort_values(workflow["hist"]["x_type"], inplace=True)
            scaled_rand = pixels_conversion_w_distance(self.rand_df, scalar)
            ax.set_title(f'{workflow["hist"]["title"]} (Rand)')
            cm = sns.color_palette(self.r_pal_type.currentText(), as_cmap=True)
            hist_df = scaled_rand[workflow["hist"]["x_type"]]
        # draw hist
        n, bins, patches = ax.hist(hist_df, bins=(int(n_bins) if n_bins.isdecimal() else n_bins), color='green')
        ax.set_xlabel(f'{workflow["hist"]["x_label"]} ({enum_to_unit(output_unit)})')
        ax.set_ylabel(workflow["hist"]["y_label"])
        # generate palette
        palette = create_color_pal(n_bins=int(len(n)), palette_type=self.pal_type.currentText())
        r_palette = create_color_pal(n_bins=int(len(n)), palette_type=self.r_pal_type.currentText())
        # normalize values
        col = (n - n.min()) / (n.max() - n.min())
        for c, p in zip(col, patches):
            p.set_facecolor(cm(c))
        # draw on canvas
        canvas.draw()
        # determine shape of canvas
        size = canvas.size()
        width, height = size.width(), size.height()
        # set hist to image of plotted hist
        self.hist = QImage(canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
        # display img
        pixmap = QPixmap.fromImage(self.hist)
        smaller_pixmap = pixmap.scaled(300, 250, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.hist_frame.setPixmap(smaller_pixmap)
        # load in image
        drawn_img = cv2.imread(self.img_drop.currentText())

        if workflow["type"] == Workflow.NND:
            # if real coords selected, annotate them on img with lines indicating length
            if self.gen_real_cb.isChecked():
                drawn_img = draw_length(nnd_df=self.real_df, bin_counts=n, img=drawn_img, palette=palette,
                                        input_unit=input_unit,
                                        scalar=scalar, circle_c=(103, 114, 0))
            # if rand coords selected, annotate them on img with lines indicating length
            if self.gen_rand_cb.isChecked():
                drawn_img = draw_length(nnd_df=self.rand_df, bin_counts=n, img=drawn_img, palette=r_palette,
                                        input_unit=input_unit,
                                        scalar=1, circle_c=(18, 156, 232))
        elif workflow["type"] == Workflow.CLUST:
            if self.gen_real_cb.isChecked():
                drawn_img = draw_clust(cluster_df=self.real_df, img=drawn_img, palette=palette, scalar=scalar)
            if self.gen_rand_cb.isChecked():
                drawn_img = draw_clust(cluster_df=self.rand_df, img=drawn_img, palette=r_palette, scalar=scalar)
        # set display img to annotated image
        self.display_img = QImage(drawn_img.data, drawn_img.shape[1], drawn_img.shape[0],
                                  QImage.Format_RGB888).rgbSwapped()
        # resize to fit on gui
        pixmap = QPixmap.fromImage(self.display_img)
        smaller_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.image_frame.setPixmap(smaller_pixmap)

    """ OPEN IMAGE IN VIEWER """
    def open_large(self, event, img):
        try:
            self.image_viewer = QImageViewer(img)
            self.image_viewer.show()
        except Exception as e:
            print(e)