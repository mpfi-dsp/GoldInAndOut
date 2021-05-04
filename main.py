# QT5
from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QTextEdit, QAction, QFileDialog, QApplication,
                             QSpacerItem, QDialog, QRadioButton, QCheckBox, QHBoxLayout, QGraphicsColorizeEffect,
                             QPushButton, QWidget, QGridLayout, QSizePolicy, QFormLayout, QLineEdit, QColorDialog)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import (QSize, Qt, QRect, QCoreApplication, QMetaObject)

# general
from pathlib import Path
from functools import partial
import sys

from colorthief import ColorThief

# main
class PipeLineGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('logo.jpg'))
        self.setMinimumSize(QSize(600, 600))
        self.setWindowTitle('MPFI EM Core Pipeline')

        layout = QFormLayout()

        # header
        self.header = QLabel("Gold Cluster Analysis For Freeze Fracture")
        self.header.setStyleSheet("font-size: 24px; font-weight: bold; padding-top: 8px; ")
        layout.addRow(self.header)
        self.desc = QLabel(
            "Simply upload the appropriate files, check the workflows you'd like to run, and click \"Start\"!")
        self.desc.setStyleSheet("font-size: 17px; padding-top: 3px; padding-bottom: 20px;")
        self.desc.setWordWrap(True)
        layout.addRow(self.desc)

        # upload header
        self.upload_header = QLabel("Upload Files")
        self.upload_header.setStyleSheet("font-size: 20px; font-weight: bold; padding-top: 8px; padding-bottom: 10px")
        layout.addRow(self.upload_header)

        # img btn
        self.img_btn = QPushButton('Upload Image', self)
        self.img_btn.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 8px; background: teal; color: white;  border-radius: 7px; ")
        self.img_btn.clicked.connect(partial(self.open_file_picker, "img"))
        # img input
        self.img_le = QLineEdit()
        self.img_le.setStyleSheet(
            "font-size: 16px; padding: 8px; font-weight: 400; background: #ddd; border-radius: 7px;")
        self.img_le.setPlaceholderText("None Selected")
        # add img row
        img_r = QHBoxLayout()
        img_r.addWidget(self.img_btn)
        img_r.addWidget(self.img_le)
        layout.addRow(img_r)

        # mask btn
        self.mask_btn = QPushButton('Upload Mask', self)
        self.mask_btn.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 8px; background: teal; color: white;  border-radius: 7px;")
        self.mask_btn.clicked.connect(partial(self.open_file_picker, "mask"))
        # mask input
        self.mask_le = QLineEdit()
        self.mask_le.setStyleSheet(
            "font-size: 16px; padding: 8px; font-weight: 400; background: #ddd; border-radius: 7px; margin-bottom: 5px;")
        self.mask_le.setPlaceholderText("None Selected")
        # mask color btn
        self.clr_btn = QPushButton('Mask Color', self)
        self.clr_btn.setStyleSheet(
            "font-size: 16px; font-weight: 500; padding: 8px; background: black; color: white;  border-radius: 7px; margin-bottom: 5px;")
        self.clr_btn.clicked.connect(self.set_mask_clr)
        # add mask row
        mask_r = QHBoxLayout()
        mask_r.addWidget(self.mask_btn)
        mask_r.addWidget(self.mask_le)
        mask_r.addWidget(self.clr_btn)
        layout.addRow(mask_r)

        # csv btn
        self.csv_btn = QPushButton('Upload CSV', self)
        self.csv_btn.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 8px; background: teal; color: white; border-radius: 7px; ")
        self.csv_btn.clicked.connect(partial(self.open_file_picker, "csv"))
        # csv input
        self.csv_le = QLineEdit()
        self.csv_le.setStyleSheet(
            "font-size: 16px; padding: 8px; font-weight: 400; background: #ddd; border-radius: 7px; margin-bottom: 5px;")
        self.csv_le.setPlaceholderText("None Selected")
        # add csv row
        csv_r = QHBoxLayout()
        csv_r.addWidget(self.csv_btn)
        csv_r.addWidget(self.csv_le)
        layout.addRow(csv_r)
        # layout.addRow(self.csv_btn, self.csv_le)

        # csv2 btn
        self.csv2_btn = QPushButton('Upload CSV2', self)
        self.csv2_btn.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 8px; background: teal; color: white; border-radius: 7px; ")
        self.csv2_btn.clicked.connect(partial(self.open_file_picker, "csv2"))
        # csv2 input
        self.csv2_le = QLineEdit()
        self.csv2_le.setStyleSheet(
            "font-size: 16px; padding: 8px; font-weight: 400; background: #ddd; border-radius: 7px; margin-bottom: 5px;")
        self.csv2_le.setPlaceholderText("None Selected")
        # add csv2 row
        csv2_r = QHBoxLayout()
        csv2_r.addWidget(self.csv2_btn)
        csv2_r.addWidget(self.csv2_le)
        layout.addRow(csv2_r)

        # workflows header
        self.workflows_header = QLabel("Select Workflows")
        self.workflows_header.setStyleSheet(
            "font-size: 20px; font-weight: bold; padding-top: 20px; padding-bottom: 10px")
        layout.addRow(self.workflows_header)

        # workflows
        self.w1_cb = QCheckBox("Annotate Gold Particles")
        self.w1_cb.setChecked(True)

        self.w2_cb = QCheckBox("K Nearest Neighbors")
        self.w2_cb.setChecked(True)

        self.w3_cb = QCheckBox("Calculate Density")
        self.w3_cb.setChecked(True)

        self.w4_cb = QCheckBox("Output Img/CSV Files")
        self.w4_cb.setChecked(True)

        # wr1 = QHBoxLayout()
        # wr1.addWidget(self.w1_cb)
        # wr1.addWidget(self.w2_cb)
        # wr1.addWidget(self.w3_cb)
        # wr1.addWidget(self.w4_cb)
        layout.addRow(self.w1_cb, self.w2_cb)
        layout.addRow(self.w3_cb, self.w4_cb)


        # start btn
        self.start_btn = QPushButton('Start', self)
        self.start_btn.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 8px; margin-top: auto; background: teal; color: white; border-radius: 7px; ")
        self.start_btn.clicked.connect(self.start)
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
                self.open_file(filename)
                self.img_le.setText(filename)
            elif btn_type == "mask":
                self.open_file(file)
                self.mask_le.setText(filename)
                # get dominant color in layer mask and assign to btn bg
                palette = ColorThief(filename)
                (r, g, b) = palette.get_color(quality=1)
                def clamp(x):
                    return max(0, min(x, 255))
                # print((r, g, b), "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b)))
                self.clr_btn.setStyleSheet(f'QWidget {{background-color: {"#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))}; font-size: 16px; font-weight: 600; padding: 8px; color: white; border-radius: 7px; }}')

            elif btn_type == "csv":
                self.open_file(file)
                self.csv_le.setText(filename)
            elif btn_type == "csv2":
                self.open_file(file)
                self.csv2_le.setText(filename)

    # open actual file
    def open_file(self, file):
        print(file)
        # if file[0]:
        #     with open(file[0], 'r') as f:
        #         data = f.read()
        #         self.textEdit.setText(data)

    def set_mask_clr(self):
        color = QColorDialog.getColor()
        print(color)
        self.clr_btn.setStyleSheet(
            'font-size: 16px; font-weight: 600; padding: 8px; background: blue; color: white; border-radius: 7px; ')
        graphic = QGraphicsColorizeEffect(self)
        graphic.setColor(color)
        self.clr_btn.setGraphicsEffect(graphic)

    def start(self):
        print(
            f'\nImg File: {self.img_le.text()}, \nMask File: {self.mask_le.text()}, \nCSV File: {self.csv_le.text()}, \nCSV2 File: {self.csv2_le.text()}')
        print(
            f'\nWorkflow 1: {self.w1_cb.isChecked()}, \nWorkflow 2: {self.w2_cb.isChecked()}, \norkflow 3: {self.w3_cb.isChecked()}, \nWorkflow 4: {self.w4_cb.isChecked()}')



styles = '''
QCheckBox {
    margin-right: 5px;
    spacing: 5px;
    font-size: 18px;    
}

QCheckBox::indicator {
    width:  27px;
    height: 27px;
}
'''


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("fusion")                 # +++
    app.setStyleSheet(styles)
    window = PipeLineGUI()
    window.show()
    sys.exit(app.exec_())
