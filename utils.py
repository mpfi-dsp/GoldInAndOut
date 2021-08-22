from PyQt5.QtCore import QThread, pyqtSignal
from PIL import Image
from typings import Unit, Workflow
from typing import List, Tuple
import seaborn as sns
import numpy as np
import pandas as pd
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


def pixels_conversion(data, unit=Unit.PIXEL, scalar=1, r=5):
    """ UPLOAD CSV AND CONVERT DF FROM ONE METRIC UNIT TO ANOTHER """
    ignored_cols = ['cluster_id', 'cluster_size']
    i = 0
    for col in data.drop(index=data.index[0], columns=data.columns[0]):
        i += 1
        print(data.columns[i])
        if data.columns[i] not in ignored_cols:
            # print(data[col].head())
            if type(data[col][0]) == tuple:
                new_col = []
                for tup in data[col]:
                    if unit == Unit.PIXEL:
                        new_col.append(tuple([round((x * scalar), r) for x in tup]))
                    else:
                        new_col.append(tuple([round((x / scalar), r) for x in tup]))
                data[col] = new_col
            else:
                if unit == Unit.PIXEL:
                    data[col] = round((data[col] * scalar), r)
                else:
                    data[col] = round(data[col].div(scalar), r)


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


def to_coord_list(df: pd.DataFrame) -> List[Tuple[float, float]]:
    # turn df into coordinate list
    x_coordinates = np.array(df['X'])
    y_coordinates = np.array(df['Y'])
    coords = []
    for i in range(len(x_coordinates)):
        coords.append([float(y_coordinates[i]), float(x_coordinates[i])])
    return coords


def to_df(coords: List[Tuple[float, float]]) -> pd.DataFrame:
    # turn coordinate list into df
    x_coords = []
    y_coords = []
    for coord in coords:
        x_coords.append(coord[1])
        y_coords.append(coord[0])
    df = pd.DataFrame(data={'X': x_coords, 'Y': y_coords})
    return df

# """ TURN ENUM INTO WORKFLOW NAME """
# def enum_to_workflow(val):
#     if val == Workflow.NND:
#         return "nnd"
#     elif val == Workflow.CLUST:
#         return "clust"
#     else:
#         return 'undefined'
