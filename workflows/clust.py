import logging
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import cv2
from utils import create_color_pal, to_df
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QImage, QColor
from typing import List, Tuple
from globals import REAL_COLOR


def run_clust(pb: pyqtSignal, real_coords: List[Tuple[float, float]], rand_coords: List[Tuple[float, float]], img_path: str, distance_threshold: int = 27, affinity: str = 'euclidean', linkage: str = 'single', clust_area: bool = False):
    """
    HIERARCHICAL CLUSTERING
    _______________________________
    @pb: progress bar wrapper element, allows us to track how much time is left in process
    @distance_threshold: using a distance threshold to automatically cluster particles
    @affinity: metric used to calc linkage
    @linkage: linkage criteria to use - determines which distance to use between sets of observation
        @single: uses the minimum of the distances between all observations of the two sets
        @ward: minimizes the variance of the clusters being merged
        @average: uses the average of the distances of each observation of the two sets
        @maximum: linkage uses the maximum distances between all observations of the two sets
    @real_coords: the real coordinates
    @rand_coords: list of randomly generated coordinates
    """
    logging.info("clustering")
    pb.emit(10)
    # handle ugly pyqt5 props
    n_clusters = None
    # cluster
    hc = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold*2, affinity=affinity, linkage=linkage)
    cluster = hc.fit_predict(real_coords)

    df = to_df(real_coords)
    df['cluster_id'] = cluster
    # random coords
    pb.emit(30)
    rand_coordinates = np.array(rand_coords)
    rand_coordinates = np.flip(rand_coordinates, 1)
    rand_cluster = hc.fit_predict(rand_coordinates)
    rand_df = pd.DataFrame(rand_coordinates, columns=["X", "Y"])
    rand_df['cluster_id'] = rand_cluster
    pb.emit(50)
    clust_details_dfs = []
    if clust_area:
        # create copy of og image
        img_og = cv2.imread(img_path)
        lower_bound = np.array([0, 250, 0])
        upper_bound = np.array([40, 255, 40])
        # iterate through clusters and find cluster area
        prog_e = 0
        for data in [df, rand_df]:
            clust_objs = []
            unique_ids = set(data['cluster_id'])
            for _id in unique_ids:
                prog_e += 20 / len(unique_ids)
                pb.emit(50 + prog_e)
                count = np.count_nonzero(np.array(data['cluster_id']) == _id)
                clust_obj = [_id, count, 0]  # id, size, area
                # create new blank image to perform calculations on
                new_img = np.zeros(img_og.shape, dtype=np.uint8)
                new_img.fill(255)
                # for each coordinate in cluster, draw circle and find contours to determine cluster area
                for index, row in data[data['cluster_id'] == _id].iterrows():
                    particle = tuple(int(x) for x in [row['X'], row['Y']])
                    new_img = cv2.circle(new_img, particle, radius=distance_threshold, color=(0, 255, 0), thickness=-1)  # thickness =  -1 for filled circle
                img_mask = cv2.inRange(new_img, lower_bound, upper_bound)
                clust_cnts, clust_hierarchy = cv2.findContours(
                    img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
                for cnt in clust_cnts:
                    area = cv2.contourArea(cnt)
                    clust_obj[2] += area
                clust_objs.append(clust_obj)
            new_df = pd.DataFrame(clust_objs, columns=["cluster_id", "cluster_size", "cluster_area"])
            new_df = new_df.reset_index(drop=True)
            clust_details_dfs.append(new_df)
    else: 
        emp_df = pd.DataFrame()
        clust_details_dfs = [emp_df, emp_df]
    pb.emit(90)
    return df, rand_df, clust_details_dfs[0], clust_details_dfs[1]


def draw_clust(clust_df: pd.DataFrame, img: List, palette: str = "rocket_r", distance_threshold: int = 27, draw_clust_area: bool = False, clust_area_color: Tuple[int, int, int] = REAL_COLOR):
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color

    if draw_clust_area:
        new_img = np.zeros(img.shape, dtype=np.uint8)
        new_img.fill(255)

    # make color pal
    palette = create_color_pal(n_bins=len(set(clust_df['cluster_id'])), palette_type=palette)
    # draw dots
    for idx, entry in clust_df.iterrows():
        particle = tuple(int(x) for x in [entry['X'], entry['Y']])
        # TODO: remove int from this next line if able to stop from converting to float
        img = cv2.circle(img, particle, 10, sea_to_rgb(palette[int(clust_df['cluster_id'][idx])]), -1)
        if draw_clust_area:
            new_img = cv2.circle(new_img, particle, radius=distance_threshold, color=(0, 255, 0), thickness=-1)
    # find centroids in df w/ clusters
    if draw_clust_area:
        lower_bound = np.array([0, 250, 0])
        upper_bound = np.array([40, 255, 40])
        clust_mask = cv2.inRange(new_img, lower_bound, upper_bound)
        clust_cnts, clust_hierarchy = cv2.findContours(clust_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        img = cv2.drawContours(img, clust_cnts, -1, clust_area_color, 3)

    def draw_clust_id_at_centroids(image, cl_df):
        for c_id in set(cl_df['cluster_id']):
            cl = cl_df.loc[cl_df['cluster_id'] == c_id]
            n, x, y = 0, 0, 0
            for idx, entry in cl.iterrows():
                x += entry['X']
                y += entry['Y']
                n += 1
            if n > 0:
                x /= n
                y /= n
                cv2.putText(image, str(int(c_id)), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=1)
    draw_clust_id_at_centroids(img, clust_df)
    return img
