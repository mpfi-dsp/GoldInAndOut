from enum import Enum

class Props(Enum):
    # general props
    CSV_SCALAR = 1
    HIST_BARS = 2
    HIST_COLORS = 3
    # nnd
    NND_GEN_RAND_COORDS = 4


class Unit(Enum):
    PIXEL = 1
    NANOMETER = 2
    MICRON = 3
    METRIC = 4

class Workflow(Enum):
    NND = 1
    CLUST = 12


class FileType(Enum):
    IMAGE = 1
    MASK = 2
    CSV = 3
    CSV2 = 4