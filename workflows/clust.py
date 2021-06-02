import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import cv2
from utils import create_color_pal

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
def run_clust(df, prog, random_coordinate_list, distance_threshold=120, n_clusters=None, affinity='euclidean', linkage='ward'):
    print("clustering")
    x_coordinates = np.array(df['X'])
    y_coordinates = np.array(df['Y'])
    real_coordinates = []
    for i in range(len(x_coordinates)):
        real_coordinates.append([float(y_coordinates[i]), float(x_coordinates[i])])

    # real coords
    prog.update_progress(30)
    real_coordinates = np.array(real_coordinates)
    print(distance_threshold, n_clusters)
    if n_clusters != "None":
        distance_threshold = None
        n_clusters = int(n_clusters)
    else:
        n_clusters = None
        distance_threshold = int(distance_threshold)
    hc = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold, affinity=affinity, linkage=linkage)
    cluster = hc.fit_predict(real_coordinates)
    df['cluster'] = cluster

    # random coords
    prog.update_progress(70)
    rand_coordinates = np.array(random_coordinate_list)
    rand_cluster = hc.fit_predict(rand_coordinates)

    rand_df = pd.DataFrame(rand_coordinates, columns=["X", "Y"])
    rand_df['cluster'] = rand_cluster

    print(rand_df.head())
    # print(cluster)
    #     # plt.figure(2)
    #     # plt.scatter(real_coordinates[:, 0], real_coordinates[:, 1], c=cluster, cmap='rainbow')
    #     # plt.savefig('./output/a.png')
    #     #
    #     # img = cv2.imread("./input/example_image.tif")
    return df, rand_df


def draw_clust(cluster_df, img, palette="rocket_r", scalar=1):
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color

    palette = create_color_pal(n_bins=len(set(cluster_df['cluster'])), palette_type=palette)

    for idx, entry in cluster_df.iterrows():
        particle = tuple(int(scalar * x) for x in [entry['X'], entry['Y']])
        img = cv2.circle(img, particle, 10, sea_to_rgb(palette[cluster_df['cluster'][idx]]), -1)
    return img
