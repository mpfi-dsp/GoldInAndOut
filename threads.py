from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QSize, QByteArray
from PyQt5.QtGui import QImage
from utils import pixels_conversion, enum_to_unit
from globals import MAX_DIRS_PRUNE
import os
import traceback
import logging
from views.logger import Logger
from typings import Unit, Workflow, DataObj, OutputOptions, WorkflowObj
from typing import List, Tuple
# workflows
from workflows.clust import run_clust, draw_clust
from workflows.gold_rippler import run_rippler, draw_rippler
from workflows.nnd_clust import run_nnd_clust, draw_nnd_clust
from workflows.starfish import run_starfish, draw_starfish
from workflows.nnd import run_nnd, draw_length
import numpy as np
import datetime
import pandas as pd
import shutil
import seaborn as sns

class AnalysisWorker(QObject):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)

    def run(self, wf: WorkflowObj, vals: List[str], coords: List[Tuple[float, float]], rand_coords: List[Tuple[float, float]], alt_coords: List[Tuple[float, float]] = None, img_path: str = "", mask_path: str = ""):
        try:
            real_df1 = real_df2 = rand_df1 = rand_df2 = pd.DataFrame()
            # ADD NEW WORKFLOWS HERE
            if wf['type'] == Workflow.NND:
                real_df1, rand_df1 = run_nnd(
                    real_coords=coords, rand_coords=rand_coords, pb=self.progress)
            elif wf['type'] == Workflow.CLUST:
                real_df1, rand_df1, real_df2, rand_df2 = run_clust(
                    real_coords=coords, rand_coords=rand_coords, img_path=img_path, distance_threshold=vals[0], n_clusters=vals[1], pb=self.progress)
            elif wf['type'] == Workflow.NND_CLUST:
                real_df2, rand_df2, real_df1, rand_df1 = run_nnd_clust(
                    real_coords=coords, rand_coords=rand_coords,  distance_threshold=vals[0],  n_clusters=vals[1], min_clust_size=vals[2], pb=self.progress)
            elif wf['type'] == Workflow.RIPPLER:
                real_df1, rand_df1 = run_rippler(real_coords=coords, alt_coords=alt_coords, rand_coords=rand_coords, pb=self.progress,
                                                 img_path=img_path, mask_path=mask_path, max_steps=vals[0], step_size=vals[1])
            elif wf['type'] == Workflow.STARFISH:
                real_df1, rand_df1 = run_starfish(
                    real_coords=coords, rand_coords=rand_coords, alt_coords=alt_coords, pb=self.progress)
            self.output_data = DataObj(real_df1, real_df2, rand_df1, rand_df2)
            self.finished.emit(self.output_data)
        except Exception as e:
            self.dlg = Logger()
            self.dlg.show()
            print(traceback.format_exc())
            logging.error(traceback.format_exc())
            self.finished.emit({})


class VisualizationsWorker(QObject):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)

    def run(self, wf: WorkflowObj, vals: List[str], coords: List[Tuple[float, float]], rand_coords: List[Tuple[float, float]], alt_coords: List[Tuple[float, float]] = None, gen_real: bool = True, gen_rand: bool = False, img_path: str = "", mask_path: str = "", selected_pal: str = "crest", selected_rand_pal: str = "rocket", n_bins = None, output_ops: OutputOptions = None, n: List[int] = np.zeros(11)):
        try:
            if gen_real or gen_rand and len(coords) > 0:
                print(f'{wf["name"]}: generating visualizations')
                # plt.close('all')
                # graph_df = pd.DataFrame([])
                # cm = plt.cm.get_cmap('crest')
                # # fig = plt.figure()
                # canvas = FigureCanvas(fig)
                # ax = fig.add_subplot(111)
                # # fix csv index not matching id
                # data.real_df1.sort_values(wf["graph"]["x_type"], inplace=True)
                # data.real_df1 = data.real_df1.reset_index(drop=True)
                # data.final_real = pixels_conversion(
                #     data=data.real_df1, unit=output_ops.output_unit, scalar=output_ops.output_scalar)
                # if wf["graph"]["x_type"] in data.rand_df1.columns and len(data.rand_df1[wf["graph"]["x_type"]]) > 0:
                #     data.rand_df1.sort_values(
                #         wf["graph"]["x_type"], inplace=True)
                #     data.rand_df1 = data.rand_df1.reset_index(drop=True)
                # if not data.rand_df1.empty:
                #     data.final_rand = pixels_conversion(
                #         data=data.rand_df1, unit=output_ops.output_unit, scalar=output_ops.output_scalar)
                # # convert back to proper size
                # if wf["graph"]["type"] == "hist":
                #     # create histogram
                #     if gen_real and not gen_rand:
                #         graph_df = data.final_real[wf["graph"]["x_type"]]
                #         cm = sns.color_palette(selected_pal, as_cmap=True)
                #         ax.set_title(f'{wf["graph"]["title"]} (Real)')
                #     elif gen_rand and not gen_real:
                #         ax.set_title(f'{wf["graph"]["title"]} (Rand)')
                #         cm = sns.color_palette(selected_rand_pal, as_cmap=True)
                #         graph_df = data.final_rand[wf["graph"]["x_type"]]
                #     if gen_real and not gen_rand or gen_rand and not gen_real:
                #         # draw graph
                #         n, bins, patches = ax.hist(graph_df, bins=(
                #             int(n_bins) if n_bins.isdecimal() else n_bins), color='green')
                #         # normalize values
                #         col = (n - n.min()) / (n.max() - n.min())
                #         for c, p in zip(col, patches):
                #             p.set_facecolor(cm(c))
                #     elif gen_real and gen_rand:
                #         rand_graph = data.final_rand[wf["graph"]["x_type"]]
                #         real_graph = data.final_real[wf["graph"]["x_type"]]
                #         ax.hist(rand_graph, bins=(int(n_bins) if n_bins.isdecimal() else n_bins), alpha=0.75,
                #                 color=create_color_pal(n_bins=1, palette_type=selected_rand_pal), label='Rand')
                #         n, bins, patches = ax.hist(real_graph, bins=(int(n_bins) if n_bins.isdecimal(
                #         ) else n_bins), alpha=0.75, color=create_color_pal(n_bins=1, palette_type=selected_pal), label='Real')
                #         ax.set_title(f'{wf["graph"]["title"]} (Real & Rand)')
                #         ax.legend(loc='upper right')
                # elif wf["graph"]["type"] == "line":
                #     # create line graph
                #     if gen_real:
                #         cm = sns.color_palette(selected_pal, as_cmap=True)
                #         ax.set_title(f'{wf["graph"]["title"]} (Real)')
                #         graph_df = data.final_real
                #     elif gen_rand:
                #         ax.set_title(f'{wf["graph"]["title"]} (Rand)')
                #         cm = sns.color_palette(selected_rand_pal, as_cmap=True)
                #         graph_df = data.final_rand
                #     ax.plot(graph_df[wf["graph"]["x_type"]],
                #             graph_df[wf["graph"]["y_type"]], color='blue')
                # elif wf["graph"]["type"] == "bar":
                #     # create bar graph
                #     if gen_real:
                #         c = 1
                #         ax.set_title(f'{wf["graph"]["title"]} (Real)')
                #         graph_y = data.final_rand[wf["graph"]["y_type"]],
                #         graph_x = np.array(
                #             data.final_real[wf["graph"]["x_type"]])
                #         # print(self.real_df[wf["graph"]["y_type"]], np.array(self.real_df[wf["graph"]["y_type"]]))
                #         if wf['type'] == Workflow.CLUST:
                #             graph_y = np.bincount(np.bincount(
                #                 data.final_real[wf["graph"]["x_type"]]))[1:]
                #             graph_x = list(range(1, (len(graph_y) + 1)))
                #             c = len(graph_x)
                #         c = create_color_pal(
                #             n_bins=c, palette_type=selected_pal)
                #         n = graph_x
                #     elif gen_rand:
                #         ax.set_title(f'{wf["graph"]["title"]} (Rand)')
                #         c = 1
                #         graph_y = data.final_rand[wf["graph"]["y_type"]],
                #         graph_x = np.array(
                #             data.final_real[wf["graph"]["x_type"]])
                #         if wf['type'] == Workflow.CLUST:
                #             graph_y = np.bincount(np.bincount(
                #                 data.final_rand[wf["graph"]["x_type"]]))[1:]
                #             graph_x = list(range(1, (len(graph_y)+1)))
                #             c = len(graph_x)
                #         c = create_color_pal(
                #             n_bins=c, palette_type=selected_rand_pal)
                #         n = graph_x
                #     if gen_real and not gen_rand or gen_rand and not gen_real:
                #         if wf['type'] == Workflow.RIPPLER:
                #             # print(graph_y[0].values)
                #             ax.bar(
                #                 graph_x, graph_y[0].values, width=20, color=c)
                #         else:
                #             # print('bar', graph_x, graph_y)
                #             bar_plot = ax.bar(graph_x, graph_y, color=c)
                #             for idx, rect in enumerate(bar_plot):
                #                 height = rect.get_height()
                #                 ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                #                         graph_y[idx],
                #                         ha='center', va='bottom', rotation=0)

                #     elif gen_real and gen_rand:
                #         real_graph_y = np.bincount(np.bincount(
                #             data.final_real[wf["graph"]["x_type"]]))[1:]
                #         real_graph_x = list(
                #             range(1, (len(set(real_graph_y)))+1))
                #         rand_graph_y = np.bincount(np.bincount(
                #             data.final_rand[wf["graph"]["x_type"]]))[1:]
                #         rand_graph_x = list(
                #             range(1, (len(set(rand_graph_y)))+1))
                #         if wf['type'] == Workflow.RIPPLER:
                #             ax.bar([el - 5 for el in np.array(data.final_rand[wf["graph"]["x_type"]])], np.array(data.final_rand[wf["graph"]
                #                                                                                                                  ["y_type"]]), width=20, alpha=0.7, color=create_color_pal(n_bins=1, palette_type=selected_rand_pal), label='Rand')
                #             ax.bar([el + 5 for el in np.array(data.final_real[wf["graph"]["x_type"]])], np.array(data.final_real[wf["graph"]
                #                                                                                                                  ["y_type"]]), width=20, alpha=0.7, color=create_color_pal(n_bins=1, palette_type=selected_pal), label='Real')
                #         else:
                #             ax.bar(rand_graph_x, rand_graph_y, color=create_color_pal(n_bins=len(
                #                 rand_graph_x), palette_type=selected_rand_pal), alpha=0.7,  label='Rand')
                #             ax.bar(real_graph_x, real_graph_y, color=create_color_pal(n_bins=len(
                #                 real_graph_x), palette_type=selected_pal),  alpha=0.7, label='Real')
                #         ax.set_title(f'{wf["graph"]["title"]} (Real & Rand)')
                #         ax.legend(loc='upper right')
                #         n = rand_graph_x

                #         # label graph
                # ax.set_xlabel(
                #     f'{wf["graph"]["x_label"]} ({enum_to_unit(output_ops.output_unit)})')
                # ax.set_ylabel(wf["graph"]["y_label"])
                # ax.set_ylim(ymin=0)
                # print(f'{wf["name"]}: generated graphs')
                # # generate palette
                # palette = create_color_pal(n_bins=int(
                #     len(n)), palette_type=selected_pal)
                # r_palette = create_color_pal(n_bins=int(
                #     len(n)), palette_type=selected_rand_pal)
                # # draw on canvas
                # canvas.draw()
                # # determine shape of canvas
                # size = canvas.size()
                # width, height = size.width(), size.height()
                # # set graph to image of plotted hist
                # self.graph = QImage(canvas.buffer_rgba(),
                #                     width, height, QImage.Format_ARGB32)
                # print(f'{wf["name"]}: generated graph')

                # load in image
                drawn_img = cv2.imread(img_path)
                # save image
                # cv2.imwrite(f'{self.img_drop.currentText()}', self.graph)
                # print(f'{wf["name"]}: saved graph')
                # TODO: ADD NEW GRAPHS HERE
                if wf["type"] == Workflow.NND:
                    # if real coords selected, annotate them on img with lines indicating length
                    if gen_real:
                        drawn_img = draw_length(
                            nnd_df=data.real_df1, bin_counts=n, img=drawn_img, palette=palette, circle_c=(103, 114, 0))
                    # if rand coords selected, annotate them on img with lines indicating length
                    if gen_rand:
                        drawn_img = draw_length(
                            nnd_df=data.rand_df1, bin_counts=n, img=drawn_img, palette=r_palette, circle_c=(18, 156, 232))
                elif wf["type"] == Workflow.CLUST:
                    if gen_real:
                        drawn_img = draw_clust(clust_df=data.real_df1, img=drawn_img,
                                               palette=palette, distance_threshold=vals[0], draw_clust_area=vals[2])
                    if gen_rand:
                        drawn_img = draw_clust(clust_df=data.rand_df1, img=drawn_img,
                                               palette=r_palette, distance_threshold=vals[0], draw_clust_area=vals[2])
                elif wf["type"] == Workflow.NND_CLUST:
                    if gen_real:
                        drawn_img = draw_nnd_clust(nnd_df=data.real_df1, clust_df=data.real_df2, img=drawn_img,
                                                   palette=palette, bin_counts=n, circle_c=(103, 114, 0), )
                    if gen_rand:
                        drawn_img = draw_nnd_clust(nnd_df=data.rand_df1, clust_df=data.rand_df2, img=drawn_img,
                                                   palette=r_palette, bin_counts=n, circle_c=(18, 156, 232), )
                elif wf["type"] == Workflow.RIPPLER:
                    if gen_real:
                        drawn_img = draw_rippler(coords=coords, alt_coords=alt_coords, mask_path=mask_path, img=drawn_img, palette=palette, circle_c=(
                            18, 156, 232), max_steps=vals[0], step_size=vals[1])
                    if gen_rand:
                        drawn_img = draw_rippler(coords=rand_coords, alt_coords=alt_coords, mask_path=mask_path,
                                                 img=drawn_img, palette=r_palette, circle_c=(103, 114, 0), max_steps=vals[0], step_size=vals[1])
                elif wf["type"] == Workflow.STARFISH:
                    # if real coords selected, annotate them on img with lines indicating length
                    if gen_real:
                        drawn_img = draw_starfish(
                            nnd_df=data.real_df1, bin_counts=n, img=drawn_img, palette=palette, circle_c=(103, 114, 0))
                    # if rand coords selected, annotate them on img with lines indicating length
                    if gen_rand:
                        drawn_img = draw_starfish(
                            nnd_df=data.rand_df1, bin_counts=n, img=drawn_img, palette=r_palette, circle_c=(18, 156, 232))
                # end graph display, set display img to annotated image
                # https://stackoverflow.com/questions/33741920/convert-opencv-3-iplimage-to-pyqt5-qimage-qpixmap-in-python
                height, width, bytesPerComponent = drawn_img.shape
                bytesPerLine = 3 * width
                cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB, drawn_img)
                self.display_img = QImage(
                    drawn_img.data, width, height, bytesPerLine, QImage.Format_RGB888)

                print(f'{wf["name"]}: finished generating visuals')
                self.finished.emit(self.display_img)
        except Exception as e:
            self.dlg = Logger()
            self.dlg.show()
            logging.error(traceback.format_exc())
            self.finished.emit()



class DownloadWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(object)
    # TODO: actually use output dir

    def run(self, wf: WorkflowObj, data: DataObj, output_ops: OutputOptions, img: str, display_img: QImage, graph: QImage):
        """ DOWNLOAD FILES """
        try:
            out_start = output_ops.output_dir if output_ops.output_dir is not None else './output'
            # delete old files to make space if applicable
            o_dir = f'{out_start}/{wf["name"].lower()}'
            if output_ops.delete_old:
                while len(os.listdir(o_dir)) >= MAX_DIRS_PRUNE:
                    oldest_dir = \
                        sorted([os.path.abspath(
                            f'{o_dir}/{f}') for f in os.listdir(o_dir)], key=os.path.getctime)[0]
                    print("pruning ", oldest_dir)
                    shutil.rmtree(oldest_dir)
                print(f'{wf["name"]}: pruned old output')
        except Exception as e:
            self.dlg = Logger()
            self.dlg.show()
            logging.error(traceback.format_exc())
            self.finished.emit()

        # download files
        try:
            print(f'{wf["name"]}: prepare to download output')
            img_name = os.path.splitext(
                os.path.basename(img))[0]
            out_dir = f'{out_start}/{wf["name"].lower()}/{img_name}-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            os.makedirs(out_dir, exist_ok=True)
            data.final_real.to_csv(f'{out_dir}/real_{wf["name"].lower()}_output_{enum_to_unit(output_ops.output_unit)}.csv',
                                   index=False, header=True)
            data.final_rand.to_csv(f'{out_dir}/rand_{wf["name"].lower()}_output_{enum_to_unit(output_ops.output_unit)}.csv',
                                   index=False, header=True)
            if display_img:
                display_img.save(
                    f'{out_dir}/drawn_{wf["name"].lower()}_img.tif')
            else:
                print(
                    'No display image generated. An error likely occurred when running workflow.')
            graph.save(f'{out_dir}/{wf["name"].lower()}_graph.jpg')
            # if workflow fills full dfs, output those two
            if not data.real_df2.empty and not data.rand_df2.empty:
                data.real_df2 = pixels_conversion(
                    data=data.real_df2, unit=output_ops.output_unit, scalar=output_ops.output_scalar)
                data.rand_df2 = pixels_conversion(
                    data=data.rand_df2, unit=output_ops.output_unit, scalar=output_ops.output_scalar)
                data.real_df2.to_csv(
                    f'{out_dir}/detailed_real_{wf["name"].lower()}_output_{enum_to_unit(output_ops.output_unit)}.csv', index=False,
                    header=True)
                data.rand_df2.to_csv(
                    f'{out_dir}/detailed_rand_{wf["name"].lower()}_output_{enum_to_unit(output_ops.output_unit)}.csv', index=False,
                    header=True)
            print(f'{wf["name"]}: downloaded output')
        except Exception as e:
            self.dlg = Logger()
            self.dlg.show()
            logging.error(traceback.format_exc())
            self.finished.emit()
