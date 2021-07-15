import logging
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import cv2

from utils import create_color_pal

def run_clust(df, pb, real_coords, rand_coords, distance_threshold=120, n_clusters=None, affinity='euclidean', linkage='ward'):
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
    @real_coords: the real coordinates
    @rand_coords: list of randomly generated coordinates
    """
    logging.info("clustering")
    pb.update_progress(30)
    if n_clusters != "None":
        distance_threshold = None
        n_clusters = int(n_clusters)
    else:
        n_clusters = None
        distance_threshold = int(distance_threshold)
    hc = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold, affinity=affinity, linkage=linkage)
    cluster = hc.fit_predict(real_coords)
    df['cluster_id'] = cluster

    # random coords
    pb.update_progress(70)
    rand_coordinates = np.array(rand_coords)
    rand_cluster = hc.fit_predict(rand_coordinates)
    rand_df = pd.DataFrame(rand_coordinates, columns=["X", "Y"])
    rand_df['cluster_id'] = rand_cluster

    return df, rand_df


def draw_clust(clust_df, img, palette="rocket_r"):
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color

    # make color pal
    palette = create_color_pal(n_bins=len(set(clust_df['cluster_id'])), palette_type=palette)
    # draw dots
    for idx, entry in clust_df.iterrows():
        particle = tuple(int(x) for x in [entry['X'], entry['Y']])
        # TODO: remove int from this next line if able to stop from converting to float
        img = cv2.circle(img, particle, 10, sea_to_rgb(palette[int(clust_df['cluster_id'][idx])]), -1)

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
                cv2.putText(image, str(int(c_id)), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255,255,255), fontScale=1)

    draw_clust_id_at_centroids(img, clust_df)

    return img
