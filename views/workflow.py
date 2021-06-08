# pyQT5
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtWidgets import (QLabel, QRadioButton, QCheckBox, QHBoxLayout, QPushButton, QWidget, QSizePolicy,
                             QFormLayout, QLineEdit,
                             QComboBox, QProgressBar, QToolButton, QVBoxLayout)
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
from workflows.nnd_clust import run_nnd_clust
from workflows.random import gen_random_coordinates

""" 
WORKFLOW PAGE
__________________
@scaled_df: dataframe containing csv coordinate data of gold particles scaled via scalar to proper unit
@workflow: JSON object containing the following data:
    @type: ENUM type of Workflow
    @header: string displayed as "header"
    @desc: string displayed as "description" below header
    @graph: graph metadata:
        @title: title of graph
        @x_label: x_label of graph
        @y_label: y_label of graph
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

        self.cstm_props = []
        for i in range(len(workflow['props'])):
            prop_l = QLabel(workflow['props'][i]['title'])
            prop_l.setStyleSheet("font-size: 17px; font-weight: 400;")
            prop_le = QLineEdit()
            prop_le.setPlaceholderText(workflow['props'][i]['placeholder'])
            layout.addRow(prop_l, prop_le)
            self.cstm_props.append(prop_le)

        """ REAL COORDS SECTION """
        gen_head = QLabel("Real Coordinates")
        gen_head.setStyleSheet("font-size: 17px; font-weight: 500; padding-top: 0px; padding-bottom: 0px; margin-top: 0px; margin-bottom: 0px;")
        self.gen_head_cb = QToolButton()
        self.gen_head_cb.setArrowType(Qt.DownArrow)
        self.gen_head_cb.setCursor(QCursor(Qt.PointingHandCursor))
        self.gen_head_cb.clicked.connect(self.toggle_gen_adv)
        layout.addRow(gen_head, self.gen_head_cb)
        # csv
        csv_lb = QLabel("csv")
        csv_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.csv_drop = QComboBox()
        self.csv_drop.addItems(csv)
        layout.addRow(csv_lb, self.csv_drop)
        # num bins
        bars_lb = QLabel(
            '<a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html"># hist bins</a>')
        bars_lb.setOpenExternalLinks(True)
        bars_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.bars_ip = QLineEdit()
        self.bars_ip.setPlaceholderText("10 OR [1, 2, 3, 4] OR 'fd'")
        layout.addRow(bars_lb, self.bars_ip)
        # color palette
        pal_lb = QLabel('<a href="https://seaborn.pydata.org/tutorial/color_palettes.html">color palette</a>')
        pal_lb.setOpenExternalLinks(True)
        pal_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.pal_type = QComboBox()
        self.pal_type.addItems(PALETTE_OPS)
        layout.addRow(pal_lb, self.pal_type)

        self.real_props = [csv_lb, self.csv_drop, pal_lb, self.pal_type, bars_lb, self.bars_ip]
        for prop in self.real_props:
            prop.setHidden(True)

        """ RANDOM COORDS SECTION """
        gen_rand_head = QLabel("Random Coordinates")
        gen_rand_head.setStyleSheet("font-size: 17px; font-weight: 500; padding-top: 0px; padding-bottom: 0px; margin-top: 0px; margin-bottom: 0px;")
        self.gen_rand_adv_cb = QToolButton()
        self.gen_rand_adv_cb.setArrowType(Qt.DownArrow)
        self.gen_rand_adv_cb.setCursor(QCursor(Qt.PointingHandCursor))
        self.gen_rand_adv_cb.clicked.connect(self.toggle_rand_adv)
        layout.addRow(gen_rand_head, self.gen_rand_adv_cb)
        # image path
        img_lb = QLabel("image")
        img_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.img_drop = QComboBox()
        self.img_drop.addItems(img)
        layout.addRow(img_lb, self.img_drop)  # csv
        # mask path
        mask_lb = QLabel("mask")
        mask_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.mask_drop = QComboBox()
        self.mask_drop.addItems(mask)
        layout.addRow(mask_lb, self.mask_drop)
        # palette random
        r_pal_lb = QLabel(
            '<a href="https://seaborn.pydata.org/tutorial/color_palettes.html">rand color palette</a>')
        r_pal_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        r_pal_lb.setOpenExternalLinks(True)
        self.r_pal_type = QComboBox()
        self.r_pal_type.addItems(PALETTE_OPS)
        self.r_pal_type.setCurrentText('crest')
        layout.addRow(r_pal_lb, self.r_pal_type)
        # num coords to gen
        n_coord_lb = QLabel("# of coords")
        n_coord_lb.setStyleSheet("font-size: 17px; font-weight: 400;")
        self.n_coord_ip = QLineEdit()
        self.n_coord_ip.setStyleSheet(
            "font-size: 16px; padding: 8px;  font-weight: 400; background: #ddd; border-radius: 7px;  margin-bottom: 5px; ") # max-width: 200px;
        self.n_coord_ip.setPlaceholderText("default is # in real csv")
        layout.addRow(n_coord_lb, self.n_coord_ip)
        # set adv hidden by default
        self.rand_props = [img_lb, self.img_drop, mask_lb, self.mask_drop, n_coord_lb, self.n_coord_ip,
                     self.r_pal_type, r_pal_lb]
        for prop in self.rand_props:
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
        # graph
        self.graph_frame = QLabel()
        self.graph_frame.setStyleSheet("padding-top: 3px; background: white;")
        self.graph_frame.setMaximumSize(400, 250)
        self.graph_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graph_frame.mouseDoubleClickEvent = lambda event: self.open_large(event, self.graph)
        self.graph_frame.setCursor(QCursor(Qt.PointingHandCursor))
        # container for visualizers
        self.img_cont = QHBoxLayout()
        self.img_cont.addWidget(self.image_frame)
        self.img_cont.addWidget(self.graph_frame)
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
            self.graph.save(f'./output/{workflow["name"].lower()}/{workflow["name"].lower()}_graph.jpg')
        except Exception as e:
            print(e)

    """ TOGGLE GENERAL ADV OPTIONS """

    def toggle_gen_adv(self):
        self.gen_head_cb.setArrowType(Qt.UpArrow if self.gen_head_cb.arrowType() == Qt.DownArrow else Qt.DownArrow)
        for prop in self.real_props:
            prop.setVisible(not prop.isVisible())

    """ TOGGLE RAND ADV OPTIONS """
    def toggle_rand_adv(self):
        self.gen_rand_adv_cb.setArrowType(Qt.UpArrow if self.gen_rand_adv_cb.arrowType() == Qt.DownArrow else Qt.DownArrow)
        for prop in self.rand_props:
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
            """ ADD NEW WORKFLOWS HERE """
            if workflow["type"] == Workflow.NND:
                self.real_df, self.rand_df = run_nnd(df=scaled_df, prog=prog_wrapper, random_coordinate_list=random_coords)
            elif workflow["type"] == Workflow.CLUST:
                vals = [self.cstm_props[i].text() if self.cstm_props[i].text() else workflow['props'][i]['placeholder'] for i in range(len(self.cstm_props))]
                self.real_df, self.rand_df = run_clust(df=scaled_df, random_coordinate_list=random_coords, prog=prog_wrapper, distance_threshold=vals[0], n_clusters=vals[1])
            elif workflow["type"] == Workflow.NND_CLUST:
                vals = [self.cstm_props[i].text() if self.cstm_props[i].text() else workflow['props'][i]['placeholder'] for i in range(len(self.cstm_props))]
                self.real_df, self.rand_df = run_nnd_clust(df=scaled_df, random_coordinate_list=random_coords, prog=prog_wrapper, distance_threshold=vals[0], n_clusters=vals[1], min_clust_size=vals[2])

            """ END OF ADD WORKFLOWS """
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
        graph_df = pd.DataFrame([])
        cm = plt.cm.get_cmap('crest')
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # create hist
        if self.gen_real_cb.isChecked():
            self.real_df.sort_values(workflow["graph"]["x_type"], inplace=True)
            graph_df = self.real_df[workflow["graph"]["x_type"]]
            cm = sns.color_palette(self.pal_type.currentText(), as_cmap=True)
            ax.set_title(f'{workflow["graph"]["title"]} (Real)')
        elif self.gen_rand_cb.isChecked():
            self.rand_df.sort_values(workflow["graph"]["x_type"], inplace=True)
            scaled_rand = pixels_conversion_w_distance(self.rand_df, scalar)
            ax.set_title(f'{workflow["graph"]["title"]} (Rand)')
            cm = sns.color_palette(self.r_pal_type.currentText(), as_cmap=True)
            graph_df = scaled_rand[workflow["graph"]["x_type"]]
        # draw graph
        n, bins, patches = ax.hist(graph_df, bins=(int(n_bins) if n_bins.isdecimal() else n_bins), color='green')
        ax.set_xlabel(f'{workflow["graph"]["x_label"]} ({enum_to_unit(output_unit)})')
        ax.set_ylabel(workflow["graph"]["y_label"])
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
        # set graph to image of plotted hist
        self.graph = QImage(canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
        # display img
        pixmap = QPixmap.fromImage(self.graph)
        smaller_pixmap = pixmap.scaled(300, 250, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.graph_frame.setPixmap(smaller_pixmap)
        # load in image
        drawn_img = cv2.imread(self.img_drop.currentText())
        """ ADD NEW GRAPHS HERE """

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
                drawn_img = draw_clust(cluster_df=self.real_df, img=drawn_img, palette=palette, scalar=scalar,)
            if self.gen_rand_cb.isChecked():
                drawn_img = draw_clust(cluster_df=self.rand_df, img=drawn_img, palette=r_palette, scalar=scalar,)

        """ END GRAPH DISPLAY """
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