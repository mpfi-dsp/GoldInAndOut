from enum import Enum
from typing_extensions import TypedDict
from typing import List
import pandas as pd

class Workflow(Enum):
    NND = 1
    CLUST = 2
    SEPARATION = 3
    RIPPLER = 4
    GOLDSTAR = 5
    # CLUST_AREA = 6


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


class WorkflowGraph(TypedDict):
    type: str
    title: str
    x_label: str
    y_label: str
    x_type: str


class WorkflowProps(TypedDict):
    title: str
    placeholder: str



class WorkflowObj(TypedDict):
   name: str
   type: Workflow
   header: str
   desc: str
   checked: bool
   graph: WorkflowGraph
   props: List[WorkflowProps]


class DataObj:
    real_df1: pd.DataFrame
    real_df2: pd.DataFrame
    rand_df1: pd.DataFrame
    rand_df2: pd.DataFrame
    final_real: pd.DataFrame
    final_rand: pd.DataFrame

    def __init__(self, real_df1: pd.DataFrame, real_df2: pd.DataFrame, rand_df1: pd.DataFrame, rand_df2: pd.DataFrame):
        self.real_df1 = real_df1
        self.real_df2 = real_df2
        self.rand_df1 = rand_df1
        self.rand_df2 = rand_df2
        self.final_real = pd.DataFrame()
        self.final_rand = pd.DataFrame()
    

class OutputOptions:
    output_unit: Unit
    output_scalar: str
    output_dir: str
    delete_old: bool

    def __init__(self, output_scalar: str, output_unit: Unit = Unit.PIXEL, output_dir: str = "./output", delete_old: bool = False):
        self.output_unit = output_unit
        self.output_scalar = output_scalar
        self.output_dir = output_dir
        self.delete_old = delete_old
