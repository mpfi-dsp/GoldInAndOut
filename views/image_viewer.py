from PyQt5.QtCore import Qt, QEvent, pyqtSignal, QRectF, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter, QIcon, QCursor, QPainterPath, QBrush, QColor
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QToolButton, QLineEdit, QWidget, QVBoxLayout, \
    QHBoxLayout, QFrame, QToolBar, QPushButton


class QImageViewer(QGraphicsView):
    photoClicked = pyqtSignal(QPoint)

    def __init__(self, img):
        super(QImageViewer, self).__init__()
        
        self.img = img
        
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(100, 100, 100)))
        self.setFrameShape(QFrame.NoFrame)

        self.toolbar = QToolBar()
        self.create_actions()
        self.create_menus()

        self.setWindowTitle("EM Image Viewer")
        self.setWindowIcon(QIcon('./assets/logo.jpg'))
        self.resize(800, 800)

        self.printer = QPrinter()

        self.setPhoto(QPixmap(img))

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())
        self.fitInView()

        # self.print_act.setEnabled(True)
        # self.update_actions()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(QImageViewer, self).mousePressEvent(event)

    def save(self):
        print("save file")
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save File", "output_file.png", "All Files(*);;", options=options)
        print(path)
        if path:
            self.img.save(path)

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self._photo.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self._photo.pixmap().rect())
            painter.drawPixmap(0, 0, self._photo.pixmap())

    def zoom_in(self):
        self.scale(1.25, 1.25)
        self._zoom += 1

    def zoom_out(self):
        self.scale(0.8, 0.8)
        self._zoom -= 1


    def create_actions(self):
        self.save_act = QAction("&Save...", self, shortcut="Ctrl+S", triggered=self.save)
        self.print_act = QAction("&Print...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_)
        self.exit_act = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoom_in_act = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoom_in)
        self.zoom_out_act = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoom_out)
        self.normal_size_act = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False,
                                       triggered=self.fitInView)
        # self.fit_to_window_act = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
        #                                  triggered=self.fit_to_window)

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

        file_btn = QPushButton("File")
        file_btn.setMenu(self.file_menu)
        self.file_menu.triggered.connect(lambda action: print(action.text()))

        view_btn = QPushButton("File")
        view_btn.setMenu(self.view_menu)
        self.file_menu.triggered.connect(lambda action: print(action.text()))

        self.toolbar.addWidget(file_btn)
        self.toolbar.addWidget(view_btn)




#
#
# class QImageViewer(QWidget):
#     def __init__(self, img):
#         super(QImageViewer, self).__init__()
#         self.viewer = PhotoViewer(self)
#         # 'Load image' button
#         self.viewer.setPhoto(QPixmap(img))
#         # Button to change from drag/pan to getting pixel info
#         self.btnPixInfo = QToolButton(self)
#         self.btnPixInfo.setText('Enter pixel info mode')
#         self.btnPixInfo.clicked.connect(self.pixInfo)
#         self.editPixInfo = QLineEdit(self)
#         self.editPixInfo.setReadOnly(True)
#         self.viewer.photoClicked.connect(self.photoClicked)
#         # Arrange layout
#         VBlayout = QVBoxLayout(self)
#         VBlayout.addWidget(self.viewer)
#         HBlayout = QHBoxLayout()
#         HBlayout.setAlignment(Qt.AlignLeft)
#         HBlayout.addWidget(self.btnPixInfo)
#         HBlayout.addWidget(self.editPixInfo)
#         VBlayout.addLayout(HBlayout)
#
#     def pixInfo(self):
#         self.viewer.toggleDragMode()
#
#     def photoClicked(self, pos):
#         if self.viewer.dragMode() == QGraphicsView.NoDrag:
#             self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))


# from PyQt5.QtCore import Qt, QEvent, pyqtSignal
# from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter, QIcon, QCursor, QPainterPath
# from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
# from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
#     qApp, QFileDialog, QGraphicsView, QGraphicsScene
#
# """ Image Viewer Widget """
#
# def adjust_scrollbar(scrollBar, factor):
#     scrollBar.setValue(int(factor * scrollBar.value()
#                            + ((factor - 1) * scrollBar.pageStep() / 2)))
#
#
# class QImageViewer(QGraphicsView):
#     def __init__(self, img):
#         super().__init__()
#
#         self.scene = QGraphicsScene()
#         self.setScene(self.scene)
#
#         self.canPan = True
#
#         self.img = img
#
#         self.printer = QPrinter()
#         self.scale_factor = 0.0
#
#         self._photo = QLabel()
#         self._photo.setBackgroundRole(QPalette.Base)
#         self._photo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
#         self._photo.setScaledContents(True)
#         self._photo.setAlignment(Qt.AlignCenter)
#         self._photo.setCursor(QCursor(Qt.SizeAllCursor))
#         # scroll mousewheel to zoom event
#         self._photo.installEventFilter(self)
#
#         self.scroll_area = QScrollArea()
#         self.scroll_area.setBackgroundRole(QPalette.Dark)
#         self.scroll_area.setWidget(self._photo)
#         self.scroll_area.setVisible(False)
#
#         self.setCentralWidget(self.scroll_area)
#
#         self.create_actions()
#         self.create_menus()
#
#         self.setWindowTitle("EM Image Viewer")
#         self.setWindowIcon(QIcon('../assets/logo.jpg'))
#         self.resize(800, 600)
#         self.open(img)
#
#         leftMouseButtonPressed = pyqtSignal(float, float)
#         rightMouseButtonPressed = pyqtSignal(float, float)
#         leftMouseButtonReleased = pyqtSignal(float, float)
#         rightMouseButtonReleased = pyqtSignal(float, float)
#
#     def eventFilter(self, source, event):
#         if (source == self._photo and event.type() == QEvent.Wheel):
#             if event.angleDelta().y() > 0:
#                 self.scale_image(1.25)
#             else:
#                 self.scale_image(0.8)
#             # do not propagate the event to the scroll area scrollbars
#             return True
#         elif event.type() == QEvent.GraphicsSceneMousePress:
#             print("press mousewheel")
#         return super().eventFilter(source, event)
#
#     def save(self):
#         print("save file")
#         options = QFileDialog.Options()
#         # options |= QFileDialog.DontUseNativeDialog
#         path, _ = QFileDialog.getSaveFileName(self, "Save File", "output_file.png", "All Files(*);;", options=options)
#         print(path)
#         if path:
#             self.img.save(path)
#
#     def open(self, img):
#         if img.isNull():
#             QMessageBox.information(self, "Image Viewer", "Cannot load image")
#             return
#         self._photo.setPixmap(QPixmap.fromImage(img))
#         # self.imageLabel.setPixmap(pixmap)
#         self.scale_factor = 1.0
#
#         self.scroll_area.setVisible(True)
#         self.print_act.setEnabled(True)
#         self.fit_to_window_act.setEnabled(True)
#         self.update_actions()
#
#         if not self.fit_to_window_act.isChecked():
#             self._photo.adjustSize()
#
#     def print_(self):
#         dialog = QPrintDialog(self.printer, self)
#         if dialog.exec_():
#             painter = QPainter(self.printer)
#             rect = painter.viewport()
#             size = self._photo.pixmap().size()
#             size.scale(rect.size(), Qt.KeepAspectRatio)
#             painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
#             painter.setWindow(self._photo.pixmap().rect())
#             painter.drawPixmap(0, 0, self._photo.pixmap())
#
#
#
#     def zoom_in(self):
#         self.scale_image(1.25)
#
#     def zoom_out(self):
#         self.scale_image(0.8)
#
#     def resize_image(self):
#         size = self.pixmap.size()
#         print(size)
#         scaled_pixmap = self.pixmap.scaled(self.scale_factor * size)
#         self._photo.setPixmap(scaled_pixmap)
#
#     def normal_size(self):
#         self._photo.adjustSize()
#         self.scale_factor = 1.0
#
#     def fit_to_window(self):
#         fit_to_window = self.fit_to_window_act.isChecked()
#         self.scroll_area.setWidgetResizable(fit_to_window)
#         if not fit_to_window:
#             self.normal_size()
#
#         self.update_actions()
#
#     def create_actions(self):
#         self.save_act = QAction("&Save...", self, shortcut="Ctrl+S", triggered=self.save)
#         self.print_act = QAction("&Print...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_)
#         self.exit_act = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
#         self.zoom_in_act = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoom_in)
#         self.zoom_out_act = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoom_out)
#         self.normal_size_act = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False,
#                                        triggered=self.normal_size)
#         self.fit_to_window_act = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
#                                          triggered=self.fit_to_window)
#
#     def create_menus(self):
#         self.file_menu = QMenu("&File", self)
#         self.file_menu.addAction(self.save_act)
#         self.file_menu.addAction(self.print_act)
#         self.file_menu.addSeparator()
#         self.file_menu.addAction(self.exit_act)
#
#         self.view_menu = QMenu("&View", self)
#         self.view_menu.addAction(self.zoom_in_act)
#         self.view_menu.addAction(self.zoom_out_act)
#         self.view_menu.addAction(self.normal_size_act)
#         self.view_menu.addSeparator()
#         self.view_menu.addAction(self.fit_to_window_act)
#
#         self.menuBar().addMenu(self.file_menu)
#         self.menuBar().addMenu(self.view_menu)
#
#     def update_actions(self):
#         self.zoom_in_act.setEnabled(not self.fit_to_window_act.isChecked())
#         self.zoom_out_act.setEnabled(not self.fit_to_window_act.isChecked())
#         self.normal_size_act.setEnabled(not self.fit_to_window_act.isChecked())
#
#     def scale_image(self, factor):
#         self.scale_factor *= factor
#         self._photo.resize(self.scale_factor * self._photo.pixmap().size())
#
#         adjust_scrollbar(self.scroll_area.horizontalScrollBar(), factor)
#         adjust_scrollbar(self.scroll_area.verticalScrollBar(), factor)
#
#         self.zoom_in_act.setEnabled(self.scale_factor < 3.0)
#         self.zoom_out_act.setEnabled(self.scale_factor > 0.333)
#
#
#     def mousePressEvent(self, event):
#         """ Start mouse pan or zoom mode.
#         """
#         scenePos = self.mapToScene(event.pos())
#         if event.button() == Qt.LeftButton:
#             if self.canPan:
#                 self.setDragMode(QGraphicsView.ScrollHandDrag)
#             self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())
#         QGraphicsView.mousePressEvent(self, event)
#
#     def mouseReleaseEvent(self, event):
#         """ Stop mouse pan or zoom mode (apply zoom if valid).
#         """
#         QGraphicsView.mouseReleaseEvent(self, event)
#         scenePos = self.mapToScene(event.pos())
#         if event.button() == Qt.LeftButton:
#             self.setDragMode(QGraphicsView.NoDrag)
#             self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
#
