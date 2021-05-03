from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QTextEdit, QAction, QFileDialog, QApplication,
                             QPushButton, QWidget)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import (QSize, Qt)
from pathlib import Path
from functools import partial
import sys



# main class
class PipeLineGUI(QWidget):
    # init gui & add ui
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowIcon(QIcon('logo.jpg'))
        self.setMinimumSize(QSize(1000, 500))
        self.setWindowTitle('MPFI EM Pipeline')

        layout = QVBoxLayout()

        # img btn
        self.img_btn = QPushButton('Upload Image', self)
        self.img_btn.setStyleSheet("font-size: 20px; font-weight: bold; background: #ddd; border-radius: 7px; ")
        self.img_btn.clicked.connect(partial(self.open_file_picker, "img"))
        # self.img_btn.resize(900, 50)
        # self.img_btn.move(50, 60)

        # mask btn
        self.mask_btn = QPushButton('Upload Mask', self)
        self.mask_btn.setStyleSheet("font-size: 20px; font-weight: bold; background: #ddd; border-radius: 7px; ")
        self.mask_btn.clicked.connect(partial(self.open_file_picker, "mask"))
        # self.mask_btn.resize(900, 50)
        # self.mask_btn.move(50, 140)

        # csv btn
        self.csv_btn = QPushButton('Upload CSV', self)
        self.csv_btn.setStyleSheet("font-size: 20px; font-weight: bold; background: #ddd; border-radius: 7px; ")
        self.csv_btn.clicked.connect(partial(self.open_file_picker, "csv"))
        # self.csv_btn.resize(900, 50)
        # self.csv_btn.move(50, 220)

        # csv2 btn
        self.csv2_btn = QPushButton('Upload CSV2', self)
        self.csv2_btn.setStyleSheet("font-size: 20px; font-weight: bold; background: #ddd; border-radius: 7px; ")
        self.csv2_btn.clicked.connect(partial(self.open_file_picker, "csv2"))
        self.csv2_btn.resize(900, 60)
        # self.csv2_btn.move(50, 300)

        layout.addWidget(self.img_btn)
        layout.addWidget(self.mask_btn)
        layout.addWidget(self.csv_btn)
        layout.addWidget(self.csv2_btn)
        self.setLayout(layout)

        # next btn
        # self.next_btn = QPushButton('Next', self)
        # self.next_btn.setStyleSheet("font-size: 20px; font-weight: bold; background: #ddd; border-radius: 7px; ")
        # # self.next_btn.clicked.connect()
        # self.verticalLayout.addWidget(self.next_btn, alignment=Qt.AlignRight)

    # open file picker and print file names
    def open_file_picker(self, btn_type):
        root_dir = str(Path.home())
        file = QFileDialog.getOpenFileName(self, 'Open file', root_dir)
        filename = file[0]
        if (len(filename)) > 0:
            if btn_type == "img":
                self.open_file(filename)
                self.img_btn.setText("IMG: " + filename)
                self.img_btn.setStyleSheet("font-size: 20px; background: #ddd; border-radius: 7px; color: teal;")
            elif btn_type == "mask":
                self.open_file(file)
                self.mask_btn.setText("MASK: " + filename)
                self.mask_btn.setStyleSheet("font-size: 20px; background: #ddd; border-radius: 7px; color: teal;")
            elif btn_type == "csv":
                self.open_file(file)
                self.csv_btn.setText("CSV: " + filename)
                self.csv_btn.setStyleSheet("font-size: 20px; background: #ddd; border-radius: 7px; color: teal;")
            elif btn_type == "csv2":
                self.open_file(file)
                self.csv2_btn.setText("CSV2: " + filename)
                self.csv2_btn.setStyleSheet("font-size: 20px; background: #ddd; border-radius: 7px; color: teal;")


    # open actual file
    def open_file(self, file):
        print(file)
        # if file[0]:
        #     with open(file[0], 'r') as f:
        #         data = f.read()
        #         self.textEdit.setText(data)


# launch gui
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PipeLineGUI()
    gui.show()
    sys.exit(app.exec_())
