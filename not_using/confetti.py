from PyQt5.QtCore import Qt, QSize, QRect, QEventLoop, QTimer, QCoreApplication
from PyQt5.QtGui import QIcon, QCursor, QMovie
from PyQt5.QtWidgets import (QWidget, QListWidget, QStackedWidget, QHBoxLayout, QListWidgetItem, QApplication, QLabel,
                             QMainWindow)

class SuccessGif(QMainWindow):
    def __init__(self):
        super().__init__()

    def mainUI(self, FrontWindow):
        FrontWindow.setObjectName("Success!")
        centralwidget = QWidget(FrontWindow)
        centralwidget.setObjectName("main-widget")
        centralwidget.setStyleSheet("background: #00ACB8;")

        # Label Create
        self.label = QLabel(centralwidget)
        self.label.setGeometry(QRect(25, 25, 200, 200))
        self.label.setMinimumSize(QSize(500, 500))
        self.label.setMaximumSize(QSize(500, 500))
        self.label.setObjectName("lb1")
        self.label.setFixedSize(500, 500)
        self.label.setAlignment(Qt.AlignCenter)
        FrontWindow.setCentralWidget(centralwidget)

        # Loading the GIF
        self.movie = QMovie("./assets/ring2.gif")
        self.label.setMovie(self.movie)
        self.movie.setScaledSize(QSize(500, 500))

        self.start_animation()

    def start_animation(self):
        self.movie.start()

