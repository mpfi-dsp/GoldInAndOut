import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd
from PIL import Image
import seaborn as sns
from typings import Unit, Workflow
import io

class Progress(QThread):
    """ PROGRESS BAR/THREADING  """
    prog = pyqtSignal(int)

    def update_progress(self, count):
        self.prog.emit(count)


def create_color_pal(n_bins=10, palette_type="crest"):
    """ GENERATE COLOR PALETTE USING SEABORN """
    palette = sns.color_palette(palette_type, n_colors=n_bins)
    color_palette = []
    for i in range(n_bins):
        color = palette[i]
        for value in color:
            value *= 255
        color_palette.append(color)
    color_palette.reverse()
    return color_palette


def get_complimentary_color(hexcode):
    """ GENERATE COMPLIMENTARY COLOR PALETTE """
    color = int(hexcode[1:], 16)
    comp_color = 0xFFFFFF ^ color
    comp_color = "#%06X" % comp_color
    return comp_color


def figure_to_img(fig):
    """ CONVERT FIGURE TO IMG """
    buf = io.BytesIO()
    # convert Matplotlib figure to PIL Image
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def pixels_conversion(csv_path, input_unit=Unit.PIXEL, csv_scalar=1, round=5):
    """ UPLOAD CSV AND CONVERT DF FROM ONE METRIC UNIT TO ANOTHER """
    data = pd.read_csv(csv_path, sep=",")
    if input_unit == Unit.PIXEL:
        data['X'] = data['X'].div(csv_scalar).round(round)
        data['Y'] = data['Y'].div(csv_scalar).round(round)
    else:
        data['X'] = (data['X'] * csv_scalar).round(round)
        data['Y'] = (data['Y'] * csv_scalar).round(round)
    return data


def pixels_conversion_w_distance(data, scalar=1):
    """ CONVERT DF FROM ONE METRIC UNIT TO ANOTHER INCLUDING DISTANCE """
    scaled_data = data.copy()
    if scalar > 1:
        for idx, entry in scaled_data.iterrows():
            scaled_data.at[idx, 'og_coord'] = tuple(int(x / scalar) for x in entry['og_coord'])
            scaled_data.at[idx, 'closest_coord'] = tuple(int(x / scalar) for x in entry['closest_coord'])
            scaled_data.at[idx, 'dist'] = float(entry['dist'] / scalar)
    return scaled_data


def unit_to_enum(val):
    """ TURN UNIT STRING INTO ENUM """
    if val == 'px':
        return Unit.PIXEL
    elif val == 'nm':
        return Unit.NANOMETER
    elif val == 'μm':
        return Unit.MICRON
    elif val == 'metric':
        return Unit.METRIC


def enum_to_unit(val):
    """ TURN ENUM INTO UNIT STRING """
    if val == Unit.PIXEL:
        return 'px'
    elif val == Unit.NANOMETER:
        return 'nm'
    elif val == Unit.MICRON:
        return 'μm'
    elif val == Unit.METRIC:
        return 'metric'
    else:
        return 'undefined'


def to_coord_list(df):
    # turn into coordinate list
    x_coordinates = np.array(df['X'])
    y_coordinates = np.array(df['Y'])
    coords = []
    for i in range(len(x_coordinates)):
        coords.append([float(y_coordinates[i]), float(x_coordinates[i])])
    return coords

# """ TURN ENUM INTO WORKFLOW NAME """
# def enum_to_workflow(val):
#     if val == Workflow.NND:
#         return "nnd"
#     elif val == Workflow.CLUST:
#         return "clust"
#     else:
#         return 'undefined'