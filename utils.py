from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from PIL import Image
import seaborn as sns
import io


""" PROGRESS BAR/THREADING  """
class Progress(QThread):
    # progress bar threading class
    prog = pyqtSignal(int)

    def update_progress(self, count):
        self.prog.emit(count)


""" COLORS/THEMING """
def create_color_pal(n_bins=10, palette_type="crest"):
    # create color palette using Seaborn
    palette = sns.color_palette(palette_type, n_colors=n_bins)
    print(palette)
    color_palette = []
    for i in range(n_bins):
        color = palette[i]
        for value in color:
            value *= 255
        color_palette.append(color)
    return color_palette


def get_complimentary_color(hexcode):
    # generate complimentary color
    color = int(hexcode[1:], 16)
    comp_color = 0xFFFFFF ^ color
    comp_color = "#%06X" % comp_color
    return comp_color


""" GENERAL UTILS """
def download_csv(df, file_name):
    # attempt to save DF as csv
    if df.shape[0] > 0 and df.shape[1] > 0:
        try:
            df.to_csv(f'./output/{file_name}', index=False, header=True)
        except Exception as e:
            print(e)


def fig2img(fig):
    # convert Matplotlib figure to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


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


""" SKLEARN """
# TODO: work on this
def run_knn(X_train, y_train, k=5):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    return k