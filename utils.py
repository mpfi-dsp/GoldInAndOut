from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from PIL import Image
import seaborn as sns
import io
import numpy as np
from typings import Unit

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


def pixels_conversion(csv_path, input_unit='px', csv_scalar=1, round=3):
    data = pd.read_csv(csv_path, sep=",")
    if input_unit == 'px':
        data['X'] = data['X'].div(csv_scalar).round(round)
        data['Y'] = data['Y'].div(csv_scalar).round(round)
    else:
        data['X'] = (data['X'] * csv_scalar).round(round)
        data['Y'] = (data['Y'] * csv_scalar).round(round)

    return data
    # x_coordinates = np.array(data[1][1:])
    # y_coordinates = np.array(data[2][1:])
    #
    # real_coordinates = []
    # for i in range(len(x_coordinates)):
    #     x = 0
    #     y = 0
    #     if input_unit == Unit.PIXEL:
    #         x = round(float(x_coordinates[i]) / csv_scalar)
    #         y = round(float(y_coordinates[i]) / csv_scalar)
    #     elif input_unit == Unit.NANOMETER or input_unit == Unit.MICRON or input_unit == Unit.METRIC:
    #         x = round(float(x_coordinates[i]) * csv_scalar)
    #         y = round(float(y_coordinates[i]) * csv_scalar)
    #     real_coordinates.append((y, x))




""" SKLEARN """
# TODO: work on this
def run_knn(X_train, y_train, k=5):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    return k