from enum import Enum

class Workflow(Enum):
    NND = 1
    CLUST = 2
    NND_CLUST = 3

class Unit(Enum):
    PIXEL = 1
    NANOMETER = 2
    MICRON = 3
    METRIC = 4

class FileType(Enum):
    IMAGE = 1
    MASK = 2
    CSV = 3
    CSV2 = 4