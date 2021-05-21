from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter, QIcon
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog

""" Image Viewer Widget """

def adjust_scrollbar(scrollBar, factor):
    scrollBar.setValue(int(factor * scrollBar.value()
                           + ((factor - 1) * scrollBar.pageStep() / 2)))


class QImageViewer(QMainWindow):
    def __init__(self, img):
        super().__init__()

        self.img = img

        self.printer = QPrinter()
        self.scale_factor = 0.0

        self.image_lb = QLabel()
        self.image_lb.setBackgroundRole(QPalette.Base)
        self.image_lb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_lb.setScaledContents(True)

        self.scroll_area = QScrollArea()
        self.scroll_area.setBackgroundRole(QPalette.Dark)
        self.scroll_area.setWidget(self.image_lb)
        self.scroll_area.setVisible(False)

        self.setCentralWidget(self.scroll_area)

        self.create_actions()
        self.create_menus()

        self.setWindowTitle("EM Image Viewer")
        self.setWindowIcon(QIcon('./assets/logo.jpg'))
        self.resize(800, 600)
        self.open(img)

    def save(self):
        print("save file")
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save File", "output_file.png", "All Files(*);;", options=options)
        print(path)
        if path:
            self.img.save(path)

    def open(self, img):
        if img.isNull():
            QMessageBox.information(self, "Image Viewer", "Cannot load image")
            return
        self.image_lb.setPixmap(QPixmap.fromImage(img))
        # self.imageLabel.setPixmap(pixmap)
        self.scale_factor = 1.0

        self.scroll_area.setVisible(True)
        self.print_act.setEnabled(True)
        self.fit_to_window_act.setEnabled(True)
        self.update_actions()

        if not self.fit_to_window_act.isChecked():
            self.image_lb.adjustSize()

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.image_lb.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.image_lb.pixmap().rect())
            painter.drawPixmap(0, 0, self.image_lb.pixmap())

    def zoom_in(self):
        self.scale_image(1.25)

    def zoom_out(self):
        self.scale_image(0.8)

    def resize_image(self):
        size = self.pixmap.size()
        print(size)
        scaled_pixmap = self.pixmap.scaled(self.scale_factor * size)
        self.image_lb.setPixmap(scaled_pixmap)

    def normal_size(self):
        self.image_lb.adjustSize()
        self.scale_factor = 1.0

    def fit_to_window(self):
        fit_to_window = self.fit_to_window_act.isChecked()
        self.scroll_area.setWidgetResizable(fit_to_window)
        if not fit_to_window:
            self.normal_size()

        self.update_actions()

    def create_actions(self):
        self.save_act = QAction("&Save...", self, shortcut="Ctrl+S", triggered=self.save)
        self.print_act = QAction("&Print...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_)
        self.exit_act = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoom_in_act = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoom_in)
        self.zoom_out_act = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoom_out)
        self.normal_size_act = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False,
                                       triggered=self.normal_size)
        self.fit_to_window_act = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                         triggered=self.fit_to_window)

    def create_menus(self):
        self.file_menu = QMenu("&File", self)
        self.file_menu.addAction(self.save_act)
        self.file_menu.addAction(self.print_act)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_act)

        self.view_menu = QMenu("&View", self)
        self.view_menu.addAction(self.zoom_in_act)
        self.view_menu.addAction(self.zoom_out_act)
        self.view_menu.addAction(self.normal_size_act)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.fit_to_window_act)

        self.menuBar().addMenu(self.file_menu)
        self.menuBar().addMenu(self.view_menu)

    def update_actions(self):
        self.zoom_in_act.setEnabled(not self.fit_to_window_act.isChecked())
        self.zoom_out_act.setEnabled(not self.fit_to_window_act.isChecked())
        self.normal_size_act.setEnabled(not self.fit_to_window_act.isChecked())

    def scale_image(self, factor):
        self.scale_factor *= factor
        self.image_lb.resize(self.scale_factor * self.image_lb.pixmap().size())

        adjust_scrollbar(self.scroll_area.horizontalScrollBar(), factor)
        adjust_scrollbar(self.scroll_area.verticalScrollBar(), factor)

        self.zoom_in_act.setEnabled(self.scale_factor < 3.0)
        self.zoom_out_act.setEnabled(self.scale_factor > 0.333)
