from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from PIL import Image
import seaborn as sns
import io

def run_knn(X_train, y_train, k=5):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    return k

def pixels_conversion(filename, nm=False):
    PIXEL_TO_MICRON_SCALAR = (1 / 1789.58611481976)
    PIXEL_TO_NANOMETER_SCALAR = 1

    try:
        df = pd.read_csv(filename, delimiter=",")
        df.drop(df.columns[[0, 2]], axis=1, inplace=True)
        if nm:
            df['X'] = df['X'] * PIXEL_TO_NANOMETER_SCALAR
            df['Y'] = df['Y'] * PIXEL_TO_NANOMETER_SCALAR
        else:
            df['X'] = df['X'] * PIXEL_TO_MICRON_SCALAR
            df['Y'] = df['Y'] * PIXEL_TO_MICRON_SCALAR
        df['X'] = df['X'].round(3)
        df['Y'] = df['Y'].round(3)
        print(df.head())
        return df
    except Exception as e:
        print(e)


def get_complimentary_color(hexcode):
    color = int(hexcode[1:], 16)
    comp_color = 0xFFFFFF ^ color
    comp_color = "#%06X" % comp_color
    return comp_color


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img



class Progress(QThread):
    prog = pyqtSignal(int)

    def update_progress(self, count):
        self.prog.emit(count)


def create_color_pal(h_bins=10):
    palette = sns.color_palette("crest")
    color_palette = []
    for i in range(h_bins):
        color = palette[i]
        for value in color:
            value *= 255
        color_palette.append(color)
    return color_palette