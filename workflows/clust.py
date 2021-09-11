import logging
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import cv2
from utils import create_color_pal, to_df
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QImage
from typing import List, Tuple


def run_clust(pb: pyqtSignal, real_coords: List[Tuple[float, float]], rand_coords: List[Tuple[float, float]], img_path: str, distance_threshold: int = 27, n_clusters: int = None, affinity: str = 'euclidean', linkage: str = 'single', clust_area: bool = False):
    """
    HIERARCHICAL CLUSTERING
    _______________________________
    @prog: progress bar wrapper element, allows us to track how much time is left in process
    @distance_threshold: using a distance threshold to automatically cluster particles
    @n_clusters: set number of clusters to use
    @affinity: metric used to calc linkage
    @linkage: linkage criteria to use - determines which distance to use between sets of observation
        @ward: minimizes the variance of the clusters being merged
        @average: uses the average of the distances of each observation of the two sets
        @maximum: linkage uses the maximum distances between all observations of the two sets
        @single: uses the minimum of the distances between all observations of the two sets
    @real_coords: the real coordinates
    @rand_coords: list of randomly generated coordinates
    """
    logging.info("clustering")
    pb.emit(30)
    # handle ugly pyqt5 props
    if n_clusters != "None":
        distance_threshold = None
        n_clusters = int(n_clusters)
    else:
        n_clusters = None
        distance_threshold = int(distance_threshold)
    # cluster
    hc = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold*2, affinity=affinity, linkage=linkage)
    cluster = hc.fit_predict(real_coords)

    df = to_df(real_coords)
    df['cluster_id'] = cluster
    # random coords
    pb.emit(50)
    rand_coordinates = np.array(rand_coords)
    rand_coordinates = np.flip(rand_coordinates, 1)
    rand_cluster = hc.fit_predict(rand_coordinates)
    rand_df = pd.DataFrame(rand_coordinates, columns=["X", "Y"])
    rand_df['cluster_id'] = rand_cluster

    pb.emit(60)
    clust_details_dfs = []
    # print('clust_area', clust_area)
    if clust_area:
        # create copy of og image
        img_og = cv2.imread(img_path)
        lower_bound = np.array([0, 250, 0])
        upper_bound = np.array([40, 255, 40])
        # iterate through clusters and find cluster area

        for data in [df, rand_df]:
            clust_objs = []
            for _id in set(data['cluster_id']):
                count = np.count_nonzero(np.array(data['cluster_id']) == _id)
                clust_obj = [_id, count, 0]  # id, size, area
                # create new blank image to perform calculations on
                new_img = np.zeros(img_og.shape, dtype=np.uint8)
                new_img.fill(255)
                # for each coordinate in cluster, draw circle and find countours to determine cluster area
                for index, row in data[data['cluster_id'] == _id].iterrows():
                    x, y = int(row['X']), int(row['Y'])
                    new_img = cv2.circle(new_img, (x, y), radius=distance_threshold, color=(0, 255, 0), thickness=-1)  # thickness =  -1 for filled circle
                    img_mask = cv2.inRange(new_img, lower_bound, upper_bound)
                    clust_cnts, clust_hierarchy = cv2.findContours(
                        img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
                    for cnt in clust_cnts:
                        area = cv2.contourArea(cnt)
                        clust_obj[2] += area
                # print(_id, clust_obj[2])
                clust_objs.append(clust_obj)

            new_df = pd.DataFrame(clust_objs, columns=["cluster_id", "cluster_size", "cluster_area"])
            new_df = new_df.reset_index(drop=True)
            clust_details_dfs.append(new_df)
    else: 
        emp_df = pd.DataFrame()
        clust_details_dfs = [emp_df, emp_df]
    # cv2.imwrite('test.tif', og_img)
    pb.emit(80)

    print('returning ', df, rand_df, clust_details_dfs)
    return df, rand_df, clust_details_dfs[0], clust_details_dfs[1]


def draw_clust(clust_df: pd.DataFrame, img: QImage , palette: str ="rocket_r", distance_threshold: int = 27, draw_clust_area: bool = False):
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color

    if distance_threshold != 27 and draw_clust_area:
        distance_threshold = int(distance_threshold)
    # make color pal
    palette = create_color_pal(n_bins=len(set(clust_df['cluster_id'])), palette_type=palette)
    # draw dots
    for idx, entry in clust_df.iterrows():
        particle = tuple(int(x) for x in [entry['X'], entry['Y']])
        # print(palette[int(clust_df['cluster_id'][idx])])
        # TODO: remove int from this next line if able to stop from converting to float
        img = cv2.circle(img, particle, 10, sea_to_rgb(palette[int(clust_df['cluster_id'][idx])]), -1)
        if draw_clust_area:
            img = cv2.circle(img, particle, radius=distance_threshold, color=(0, 255, 0))
            # find centroids in df w/ clusters

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
                cv2.putText(image, str(int(c_id)), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            color=(255, 255, 255), fontScale=1)

    draw_clust_id_at_centroids(img, clust_df)
    return img
