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
from workflows import random_coords

distance_threshold = 27
min_clust_size = 2

def run_clust(pb: pyqtSignal, real_coords: List[Tuple[float, float]], rand_coords: List[Tuple[float, float]], img_path: str, min_clust_size: int = 2, distance_threshold: int = 27, affinity: str = 'euclidean', linkage: str = 'single', clust_area: bool = False):
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
    @min_clust_size: minimum number of coords required to be considered a "cluster"
    """

    def minify_df(df, k: int = 2, c: int = 1): # adapted from Separation
        for el in df.iloc[:, c]:
            if el < k:
                drop_values = df[df.iloc[:, c] == el].index
                df.drop(drop_values, inplace=True)
        return df

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
    if random_coords.N > 1:
        len_real = len(real_coords)
        rand_coordinates_chunk = [rand_coordinates[i:i+len_real] for i in range(0, len(rand_coords), len_real)]
        rand_cluster_1 = [x for rand_coordinates_chunk in [hc.fit_predict(x) for x in rand_coordinates_chunk] for x in rand_coordinates_chunk]
        # updating cluster IDs to be unique for each trial
        blank = rand_cluster_1[0:len_real].copy()
        first_trial = rand_cluster_1[0:len_real].copy()
        double_len_real = len_real * 2
        for i in range(random_coords.N - 1):
            max_id = int(max(first_trial)) + 1
            testing = [x+max_id for x in rand_cluster_1[len_real:double_len_real]]
            first_trial += testing
            blank += testing
            len_real += len(real_coords)
            double_len_real += len(real_coords)
        rand_cluster = np.array(blank)
    else:
        rand_cluster = hc.fit_predict(rand_coordinates)
    rand_df = pd.DataFrame(rand_coordinates, columns=["X", "Y"])
    rand_df['cluster_id'] = rand_cluster
    len_real_2 = len(real_coords)
    rand_df_og_copy = rand_df.copy()
    if not clust_area:
        for i in range(random_coords.N):
            if len(rand_df[0:]) > len_real_2:
                bottom_rows = rand_df.loc[len_real_2:, :]
                bottom_rows = bottom_rows.reset_index(drop=True)
                A = len_real_2 * random_coords.N
                N_minus_one = random_coords.N - 1
                B = len_real_2 * N_minus_one
                to_drop = A - B
                rand_df = rand_df.head(-to_drop)
                rand_df = pd.merge(rand_df, bottom_rows, how='outer', left_index=True, right_index=True)
                rand_df = rand_df.loc[:, ~rand_df.apply(lambda x: x.duplicated(), axis=1).all()].copy()
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
            clust_details_dfs.append(new_df)
            # regular random
            if random_coords.N > 1:
                for i in range(random_coords.N):
                    if len(rand_df[0:]) > len_real_2:
                        bottom_rows = rand_df.loc[len_real_2:, :]
                        bottom_rows = bottom_rows.reset_index(drop=True)
                        A = len_real_2 * random_coords.N
                        N_minus_one = random_coords.N - 1
                        B = len_real_2 * N_minus_one
                        to_drop = A - B
                        rand_df = rand_df.head(-to_drop)
                        rand_df = pd.merge(rand_df, bottom_rows, how='outer', left_index=True, right_index=True)
                        rand_df = rand_df.loc[:, ~rand_df.apply(lambda x: x.duplicated(), axis=1).all()].copy()
    else:
        emp_df = pd.DataFrame()
        clust_details_dfs = [emp_df, emp_df]
    final_sum = pd.DataFrame()
    clust_details_dfs[0] = minify_df(df=clust_details_dfs[0], k=min_clust_size, c=1) # detailed_real_clust_output
    clust_details_dfs[1] = minify_df(df=clust_details_dfs[1], k=min_clust_size, c=1) # detailed_rand_clust_output
    clust_details_dfs[1] = clust_details_dfs[1].reset_index()
    clust_details_dfs[1] = clust_details_dfs[1].iloc[:, 1:]
    start_iteration = 0 # start of interation
    M = 0
    iteration_no = 1 # iteration number
    rand_df_og_copy = rand_df_og_copy.sort_values(by="cluster_id")
    rand_df_og_copy = rand_df_og_copy.reset_index()
    if clust_details_dfs[1].empty:
        clust_details_dfs[1] = pd.DataFrame(np.array([[0, 0, 0, 0]]), columns=['cluster_id', 'cluster_size', 'cluster_area', 'summed_trial_area'])
    elif len(clust_details_dfs[1]) > 1:
        for i in range(random_coords.N):
            # within one trial of coordinates
            rand_df_copy = rand_df_og_copy.iloc[:, 1:]
            rand_df_copy = rand_df_copy.iloc[start_iteration:(len(df) * iteration_no), :]
            rand_df_copy = rand_df_copy.reset_index()
            rand_df_copy = rand_df_copy.iloc[:, 1:]
            if M > 0:
                rand_df_copy = rand_df_copy[rand_df_copy.iloc[:, 0] >= M]
            v = rand_df_copy.iloc[:, 2].value_counts()
            # find cluster IDs that show up more than the min clust size criteria (input)
            sorted_IDs_2 = rand_df_copy[rand_df_copy.iloc[:, 2].isin(v.index[v.gt(min_clust_size - 1)])]
            # find the maximum cluster ID within the trial
            max_ID = int(max(sorted_IDs_2.iloc[:, 2]))
            # find the index of the coordinate in the centroid list
            # https://sparkbyexamples.com/pandas/get-row-number-in-pandas/
            thresh = clust_details_dfs[1][clust_details_dfs[1].iloc[:, 0] == max_ID].index.to_numpy() 
            thresh  = int(thresh[0]) + 1
            details_df_copy = clust_details_dfs[1].copy()
            insert_rows = int(thresh - start_iteration - 1)
            details_df_copy = details_df_copy.iloc[start_iteration:thresh, 2]
            summed = details_df_copy.sum(axis=0)
            summed = pd.DataFrame([summed], columns=["summed_trial_area"])
            summed = pd.concat([summed,
                pd.DataFrame(0, columns=summed.iloc[1:, 0], index=range(insert_rows))], ignore_index=True)
            final_sum = pd.concat([final_sum, summed], ignore_index=True)
            if len(final_sum) < len(clust_details_dfs[1]):
                start_iteration = int(thresh)
                iteration_no += 1
                M += int(len(df) // random_coords.N)
        clust_details_dfs[1]["summed_trial_area"] = final_sum
        clust_details_dfs[1] = clust_details_dfs[1].fillna(0)
    final_sum_real = pd.DataFrame()
    insert_rows_real = len(clust_details_dfs[0]) - 1
    real_details_dfs_copy = clust_details_dfs[0].iloc[:, 2].copy()
    summed_real = real_details_dfs_copy.sum(axis=0)
    summed_real = pd.DataFrame([summed_real], columns=["summed_area"])
    summed_real = pd.concat([summed_real,
        pd.DataFrame(0, columns=summed_real.iloc[1:, 0], index=range(insert_rows_real))], ignore_index=True)
    final_sum_real = pd.concat([final_sum_real, summed_real], ignore_index=True)
    clust_details_dfs[0]["summed_area"] = final_sum_real
    clust_details_dfs[0] = clust_details_dfs[0].fillna(0)
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
