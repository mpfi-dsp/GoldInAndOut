import logging

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QPlainTextEdit, QDialog, QPushButton, QVBoxLayout


# logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(name)s - %(levelname)s - %(message)s')

class QPlainTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.logger = QPlainTextEdit(parent)
        self.logger.setStyleSheet("background: #ddd;")
        self.logger.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.logger.appendPlainText(msg)


class Logger(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('GoldInAndOut Logger')
        self.setWindowIcon(QIcon('./assets/logo.png'))
        self.setMinimumSize(QSize(600, 300))
        # self.setMaximumSize(QSize(600, 300))

        log_text_box = QPlainTextEditLogger(self)
        # You can format what is printed to text box
        log_text_box.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_text_box)
        # You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)

        self._button = QPushButton(self)
        self._button.setText('Test Me')

        layout = QVBoxLayout()
        # Add the new logging box widget to the layout
        layout.addWidget(log_text_box.logger)
        layout.addWidget(self._button)
        self.setLayout(layout)

        # Connect signal to slot
        self._button.clicked.connect(self.test)

    def test(self):
        logging.debug('damn, a bug')
        logging.info('something to remember')
        logging.warning('that\'s not right')
        logging.error('foobar')