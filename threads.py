from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QSize, QByteArray
from PyQt5.QtGui import QImage
from utils import pixels_conversion, enum_to_unit, to_coord_list
from globals import MAX_DIRS_PRUNE
import os
import traceback
import logging
from views.logger import Logger
from views import home
from typings import Unit, Workflow, DataObj, OutputOptions, WorkflowObj
from typing import List, Tuple
# workflows
from workflows.clust import run_clust, distance_threshold, min_clust_size
from workflows.gold_rippler import run_rippler
from workflows.separation import run_separation
from workflows.goldstar import run_goldstar
from workflows.nnd import run_nnd
from workflows.random_coords import gen_random_coordinates
from workflows import random_coords
# from workflows.astar import run_astar
from workflows.goldAstar import run_goldAstar
import numpy as np
import datetime
import pandas as pd
import shutil
from PyQt5.QtWidgets import QWidget


class DataLoadWorker(QObject):
    finished = pyqtSignal(list)

    def run(self, img_path: str = "", mask_path: str = "",  csv_path: str = "", csv2_path: str = "", unit: Unit = Unit.PIXEL, scalar: float = 1.0, parameters: float = 1.0):
        try:
            data = pd.read_csv(csv_path, sep=",")
            if len(data.columns) > 2: # if there are more than two columns; not preprocessed (assuming the first two are X, Y)
                data = data.iloc[:, 3:]
                data.columns.values[0] = 'X'
                data.columns.values[1] = 'Y'
            scaled_df = pixels_conversion(data=data, unit=unit, scalar=scalar)
            COORDS = to_coord_list(scaled_df)
    
            if len(csv2_path) > 0:
                data = pd.read_csv(csv2_path, sep=",")
                ALT_COORDS = to_coord_list(
                    pixels_conversion(data=data, unit=unit, scalar=scalar))
            elif len(csv2_path) == 0: 
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
                    real_coords=coords, rand_coords=rand_coords, img_path=img_path, distance_threshold=vals[0], min_clust_size=vals[1], pb=self.progress, clust_area=clust_area)
            elif wf['type'] == Workflow.SEPARATION:
                real_df2, rand_df2, real_df1, rand_df1 = run_separation(
                    real_coords=coords, rand_coords=rand_coords,  distance_threshold=vals[0], min_clust_size=vals[1], pb=self.progress, clust_area=clust_area)
            elif wf['type'] == Workflow.RIPPLER:
                if random_coords.N > 1:
                    n = len(coords) #chunk row size
                    random_chunks = [rand_coords[i:i+n] for i in range(0, len(rand_coords), n)]
                    random_rippler_df = pd.DataFrame()
                    index_no = 0
                    # defining real df
                    real_df1, rand_unused = run_rippler(real_coords=coords, alt_coords=alt_coords, rand_coords=random_chunks[0],
                                                  pb=self.progress, img_path=img_path, mask_path=mask_path, max_steps=vals[2], step_size=vals[3], initial_radius=vals[4])
                    # rand df
                    for i in range(random_coords.N):
                        real_unused, rand_df1 = run_rippler(real_coords=coords, alt_coords=alt_coords, rand_coords=random_chunks[index_no],
                                                  pb=self.progress, img_path=img_path, mask_path=mask_path, max_steps=vals[2], step_size=vals[3], initial_radius=vals[4])
                        blank_df = len(rand_df1[0:]) - 1
                        avg_df = pd.DataFrame(rand_df1.loc[:, (rand_df1.columns.str.startswith('L'))].mean()).transpose() 
                        zeros_df = pd.DataFrame(0, index=range(blank_df), columns=avg_df.columns)
                        clean_avg_df = pd.concat([avg_df, zeros_df], ignore_index=True, axis=0)
                        renamed = [(i, 'avg_' + i) for i in  rand_df1.columns.values]
                        clean_avg_df.rename(columns=dict(renamed), inplace=True)
                        rand_df1 = pd.merge(rand_df1, clean_avg_df, how='outer', left_index=True, right_index=True)
                        random_rippler_df = pd.concat([random_rippler_df, rand_df1])
                        if index_no <= random_coords.N - 1:
                            index_no += 1
                    rand_df1 = random_rippler_df
                else:
                    real_df1, rand_df1 = run_rippler(real_coords=coords, alt_coords=alt_coords, rand_coords=rand_coords, pb=self.progress, img_path=img_path, mask_path=mask_path, max_steps=vals[2], step_size=vals[3], initial_radius=vals[4])
            elif wf['type'] == Workflow.GOLDSTAR:
                real_df1, rand_df1 = run_goldstar(
                    real_coords=coords, rand_coords=rand_coords, alt_coords=alt_coords, pb=self.progress) #img_path=img_path, mask_path=mask_path, a_star=vals[0])
                if random_coords.N > 1:
                    rand_df1 = rand_df1.iloc[0:len(real_df1), :]
            # elif wf['type'] == Workflow.ASTAR:
            #     # real_df1, rand_df1, real_df2, rand_df2 = run_astar(
            #     real_df1, rand_df1, real_df2, rand_df2 = run_goldAstar(
            #         map_path=img_path, mask_path=mask_path, coord_list=coords, random_coord_list=rand_coords, alt_list=alt_coords, pb=self.progress)
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
                    # filter out macos system files
                    if '.DS_Store' not in oldest_dir:
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
            if random_coords.N > 1: 
                suf = 2 * (random_coords.N - 1) # removing suffix from column name 
                if [col for col in data.final_rand.columns if col.endswith('_x')] or [col for
                    col in data.final_rand.columns if col.endswith('_y')]:
                    data.final_rand.columns = pd.Index(map(lambda x: str(x)[:-suf], data.final_rand.columns)) # https://stackoverflow.com/questions/37061541/remove-last-two-characters-from-column-names-of-all-the-columns-in-dataframe-p
                    if wf['type'] == Workflow.NND or wf['type'] == Workflow.GOLDSTAR: 
                        data.final_rand.columns.values[-1] = "total_avg_dist"
            if data.final_rand.empty: 
                data.final_rand = pd.DataFrame(np.array([[0, 0, 0, 0, 0]]), columns=['og_centroid', 'closest_centroid', 'dist', 'cluster_id', 'avg_trial_dist'])
            data.final_real.to_csv(f'{out_dir}/{img_name}_real_{wf["name"].lower()}_output_{enum_to_unit(output_ops.output_unit)}-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv',
                                   index=False, header=True)
            data.final_rand.to_csv(f'{out_dir}/{img_name}_rand_{wf["name"].lower()}_output_{enum_to_unit(output_ops.output_unit)}-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv',
                                   index=False, header=True)
            data.input_params.to_csv(f'{out_dir}/{img_name}_parameters_{wf["name"].lower()}_output-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt',
                                index=False, header=True)
            if display_img:
                display_img.save(
                    f'{out_dir}/{img_name}_drawn_{wf["name"].lower()}_img-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.tif')
            else:
                logging.info(
                    'No display image generated. An error likely occurred when running workflow.')
            graph.save(f'{out_dir}/{wf["name"].lower()}_graph.jpg')
            # if workflow fills full dfs, output those two
            logging.info('attempting to save dfs')
            if wf['type'] == Workflow.CLUST or wf['type'] == Workflow.SEPARATION: 
                if len(data.real_df2) > 1: 
                    real_df2 = pixels_conversion(
                        data=data.real_df2, unit=Unit.PIXEL, scalar=float(output_ops.output_scalar))
                else: 
                    real_df2 = data.real_df2
                if random_coords.N > 1:
                    if [col for col in data.rand_df2.columns if col.endswith('_x')] or [col for
                            col in data.rand_df2.columns if col.endswith('_y')]: 
                        data.rand_df2.columns = [col[:-suf] for col in data.rand_df2.columns] # excluding total avg
                    if wf['type'] == Workflow.CLUST: 
                        rand_df2 = pixels_conversion(
                            data=data.rand_df2, unit=Unit.PIXEL, scalar=float(output_ops.output_scalar))
                        real_df2.iloc[:, -1:] = pixels_conversion(
                                data=real_df2.iloc[:, -1:], unit=Unit.PIXEL, scalar=float(output_ops.output_scalar))
                        rand_df2.iloc[:, -1:] = pixels_conversion(
                                data=rand_df2.iloc[:, -1:], unit=Unit.PIXEL, scalar=float(output_ops.output_scalar))
                    elif wf['type'] == Workflow.SEPARATION: 
                        rand_df2 = data.rand_df2
                else:
                    rand_df2 = pixels_conversion(
                        data=data.rand_df2, unit=Unit.PIXEL, scalar=float(output_ops.output_scalar))
                if wf['type'] == Workflow.CLUST:
                    if real_df2.empty:
                        real_df2 = pd.DataFrame(np.array([[0, 0, 0, 0, 0]]), columns=['cluster_id', 'cluster_size', 'cluster_area'])
                    if rand_df2.empty: 
                        rand_df2 = pd.DataFrame(np.array([[0, 0, 0, 0, 0]]), columns=['cluster_id', 'cluster_size', 'cluster_area'])
                real_df2.to_csv(
                    f'{out_dir}/{img_name}_detailed_real_{wf["name"].lower()}_output_{enum_to_unit(output_ops.output_unit)}-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False,
                    header=True)
                rand_df2.to_csv(
                    f'{out_dir}/{img_name}_detailed_rand_{wf["name"].lower()}_output_{enum_to_unit(output_ops.output_unit)}-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False,
                    header=True)
            self.finished.emit()
            logging.info("%s: downloaded output, closing thread", wf["name"])
        except Exception as e:
            self.dlg = Logger()
            self.dlg.show()
            logging.error(traceback.format_exc())
            self.finished.emit()