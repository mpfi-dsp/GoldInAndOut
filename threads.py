from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QSize, QByteArray
from PyQt5.QtGui import QImage
from utils import pixels_conversion, enum_to_unit, to_coord_list
from globals import MAX_DIRS_PRUNE
import os
import traceback
import logging
from views.logger import Logger
from typings import Unit, Workflow, DataObj, OutputOptions, WorkflowObj
from typing import List, Tuple
# workflows
from workflows.clust import run_clust
from workflows.gold_rippler import run_rippler
from workflows.separation import run_separation
from workflows.goldstar import run_goldstar
from workflows.nnd import run_nnd
from workflows.random_coords import gen_random_coordinates
import numpy as np
import datetime
import pandas as pd
import shutil


class DataLoadWorker(QObject):
    finished = pyqtSignal(list)

    def run(self, img_path: str = "", mask_path: str = "",  csv_path: str = "", csv2_path: str = "", unit: Unit = Unit.PIXEL, scalar: float = 1.0):
        try:
            data = pd.read_csv(csv_path, sep=",")
            scaled_df = pixels_conversion(data=data, unit=unit, scalar=scalar)
            COORDS = to_coord_list(scaled_df)
    
            if len(csv2_path) > 0:
                data = pd.read_csv(csv2_path, sep=",")
                ALT_COORDS = to_coord_list(
                    pixels_conversion(data=data, unit=unit, scalar=scalar))
            else:
                ALT_COORDS = gen_random_coordinates(img_path, mask_path, count=len(COORDS))
    
            self.finished.emit([COORDS, ALT_COORDS])
            logging.info("Finished loading in and converting data")
        except Exception as e:
            self.dlg = Logger()
            self.dlg.show()
            logging.error(traceback.format_exc())
            self.finished.emit([])


class AnalysisWorker(QObject):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)

    def run(self, wf: WorkflowObj, vals: List[str], coords: List[Tuple[float, float]], rand_coords: List[Tuple[float, float]], alt_coords: List[Tuple[float, float]] = None, img_path: str = "", mask_path: str = "", clust_area: bool = False):
        try:
            real_df1 = real_df2 = rand_df1 = rand_df2 = pd.DataFrame()
            print('vals', vals)
            # ADD NEW WORKFLOWS HERE
            if wf['type'] == Workflow.NND:
                real_df1, rand_df1 = run_nnd(
                    real_coords=coords, rand_coords=rand_coords, pb=self.progress)
            elif wf['type'] == Workflow.CLUST:
                real_df1, rand_df1, real_df2, rand_df2 = run_clust(
                    real_coords=coords, rand_coords=rand_coords, img_path=img_path, distance_threshold=vals[0], pb=self.progress, clust_area=clust_area)
            elif wf['type'] == Workflow.SEPARATION:
                real_df2, rand_df2, real_df1, rand_df1 = run_separation(
                    real_coords=coords, rand_coords=rand_coords,  distance_threshold=vals[0], min_clust_size=vals[1], pb=self.progress, clust_area=clust_area)
            elif wf['type'] == Workflow.RIPPLER:
                real_df1, rand_df1 = run_rippler(real_coords=coords, alt_coords=alt_coords, rand_coords=rand_coords, pb=self.progress, img_path=img_path, mask_path=mask_path, max_steps=vals[0], step_size=vals[1], initial_radius=vals[2])
            elif wf['type'] == Workflow.GOLDSTAR:
                real_df1, rand_df1 = run_goldstar(
                    real_coords=coords, rand_coords=rand_coords, alt_coords=alt_coords, pb=self.progress) #img_path=img_path, mask_path=mask_path, a_star=vals[0])
            self.output_data = DataObj(real_df1, real_df2, rand_df1, rand_df2)
            self.finished.emit(self.output_data)
            logging.info('finished %s analysis', wf["name"])
        except Exception as e:
            self.dlg = Logger()
            self.dlg.show()
            logging.error(traceback.format_exc())
            self.finished.emit({})


class DownloadWorker(QObject):
    finished = pyqtSignal()

    def run(self, wf: WorkflowObj, data: DataObj, output_ops: OutputOptions, img: str, display_img: QImage, graph: QImage):
        """ DOWNLOAD FILES """
        # logging.info(output_ops.delete_old, output_ops.output_dir, output_ops.output_scalar, output_ops.output_unit)
        try:
            out_start = output_ops.output_dir if output_ops.output_dir is not None else './output'
            # delete old files to make space if applicable
            o_dir = f'{out_start}/{wf["name"].lower()}'
            if output_ops.delete_old:
                while len(os.listdir(o_dir)) >= MAX_DIRS_PRUNE:
                    oldest_dir = \
                        sorted([os.path.abspath(
                            f'{o_dir}/{f}') for f in os.listdir(o_dir)], key=os.path.getctime)[0]
                    logging.info("pruning %s", oldest_dir)
                    shutil.rmtree(oldest_dir)
                logging.info('%s: pruned old output', wf["name"])
        except Exception as e:
            self.dlg = Logger()
            self.dlg.show()
            logging.error(traceback.format_exc())
            self.finished.emit()

        # download files
        try:
            logging.info(
                '%s: prepare to download output', wf["name"])
            img_name = os.path.splitext(
                os.path.basename(img))[0]
            out_dir = f'{out_start}/{wf["name"].lower()}/{img_name}-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            os.makedirs(out_dir, exist_ok=True)
            logging.info('attempting to save cleaned dfs')
            data.final_real.to_csv(f'{out_dir}/real_{wf["name"].lower()}_output_{enum_to_unit(output_ops.output_unit)}.csv',
                                   index=False, header=True)
            data.final_rand.to_csv(f'{out_dir}/rand_{wf["name"].lower()}_output_{enum_to_unit(output_ops.output_unit)}.csv',
                                   index=False, header=True)
            if display_img:
                display_img.save(
                    f'{out_dir}/drawn_{wf["name"].lower()}_img.tif')
            else:
                logging.info(
                    'No display image generated. An error likely occurred when running workflow.')
            graph.save(f'{out_dir}/{wf["name"].lower()}_graph.jpg')
            # if workflow fills full dfs, output those two
            logging.info('attempting to save dfs')
            if not data.real_df2.empty and not data.rand_df2.empty:
                real_df2 = pixels_conversion(
                    data=data.real_df2, unit=Unit.PIXEL, scalar=float(output_ops.output_scalar))
                rand_df2 = pixels_conversion(
                    data=data.rand_df2, unit=Unit.PIXEL, scalar=float(output_ops.output_scalar))
                real_df2.to_csv(
                    f'{out_dir}/detailed_real_{wf["name"].lower()}_output_{enum_to_unit(output_ops.output_unit)}.csv', index=False,
                    header=True)
                rand_df2.to_csv(
                    f'{out_dir}/detailed_rand_{wf["name"].lower()}_output_{enum_to_unit(output_ops.output_unit)}.csv', index=False,
                    header=True)
            self.finished.emit()
            logging.info("%s: downloaded output, closing thread", wf["name"])
        except Exception as e:
            self.dlg = Logger()
            self.dlg.show()
            logging.error(traceback.format_exc())
            self.finished.emit()
