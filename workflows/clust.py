import logging

import pandas as pd
from PIL import ImageFont, ImageDraw
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import cv2
from sortedcollections import OrderedSet

from utils import create_color_pal

def run_clust(df, pb, rand_coords, distance_threshold=120, n_clusters=None, affinity='euclidean', linkage='ward'):
    """
    WARD HIERARCHICAL CLUSTERING
    _______________________________
    @df: dataframe with coordinates scaled to whatever format desired
    @prog: progress bar wrapper element, allows us to track how much time is left in process
    @distance_threshold: using a distance threshold to automatically cluster particles
    @n_clusters: set number of clusters to use
    @affinity: metric used to calc linkage
    @linkage: linkage criteria to use - determines which distance to use between sets of observation
        @ward: minimizes the variance of the clusters being merged
        @average: uses the average of the distances of each observation of the two sets
        @maximum: linkage uses the maximum distances between all observations of the two sets
        @single: uses the minimum of the distances between all observations of the two sets
    @random_coordinate_list: list of randomly generated coordinates
    """
    logging.info("clustering")
    x_coordinates = np.array(df['X'])
    y_coordinates = np.array(df['Y'])
    real_coordinates = []
    for i in range(len(x_coordinates)):
        real_coordinates.append([float(y_coordinates[i]), float(x_coordinates[i])])
    # real coords
    pb.update_progress(30)
    real_coordinates = np.array(real_coordinates)
    # print(distance_threshold, n_clusters)
    if n_clusters != "None":
        distance_threshold = None
        n_clusters = int(n_clusters)
    else:
        n_clusters = None
        distance_threshold = int(distance_threshold)
    hc = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold, affinity=affinity, linkage=linkage)
    cluster = hc.fit_predict(real_coordinates)
    df['cluster_id'] = cluster

    temp_df = pd.DataFrame([])
    temp_df['unique_cluster_ids'] = pd.Series(OrderedSet(cluster))
    temp_df['particle_count'] = pd.Series(np.bincount(np.array(cluster)))
    print(temp_df.head())

    # random coords
    pb.update_progress(70)
    rand_coordinates = np.array(rand_coords)
    rand_cluster = hc.fit_predict(rand_coordinates)
    rand_df = pd.DataFrame(rand_coordinates, columns=["X", "Y"])
    rand_df['cluster_id'] = rand_cluster

    # print(rand_df.head())
    return df, rand_df


def draw_clust(clust_df, img, palette="rocket_r", scalar=1):
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color
    # make color pal
    palette = create_color_pal(n_bins=len(set(clust_df['cluster_id'])), palette_type=palette)
    # draw dots
    for idx, entry in clust_df.iterrows():
        particle = tuple(int(scalar * x) for x in [entry['X'], entry['Y']])
        img = cv2.circle(img, particle, 10, sea_to_rgb(palette[clust_df['cluster_id'][idx]]), -1)

    # find centroids in df w/ clusters
    def draw_clust_id_at_centroids(image, cl_df):
        centroids, centroid_ids = [], []
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
                # centroids.append((round(y, 3), round(x, 3)))
                # centroid_ids.append(c_id)
                # TODO: draw number for each cluster at centroid i.e. draw((coords, text=id))
                # draw.text((round(y, 3), round(x, 3)), str(c_id), fill=(255,255,255), font=ImageFont.truetype('Roboto-Bold.ttf', size=15))

    draw_clust_id_at_centroids(img, clust_df)

    return img
