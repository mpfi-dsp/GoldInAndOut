import logging

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from typings import Unit
from utils import create_color_pal
from collections import Counter
import math
import cv2

def run_nnd_clust(df, pb, real_coords, rand_coords, min_clust_size=3, distance_threshold=120, n_clusters=None, affinity='euclidean', linkage='ward'):
    """
    NEAREST NEIGHBOR DISTANCE OF WARD HIERARCHICAL CLUSTERING
    _______________________________
    @df: dataframe with coordinates scaled to whatever format desired
    @pb: progress bar wrapper element, allows us to track how much time is left in process
    @real_coords: list of real coordinates
    @rand_coords: list of randomly generated coordinates
    @min_clust_size: minimum number of coords required to be considered a "cluster"
    @distance_threshold: using a distance threshold to automatically cluster particles
    @n_clusters: set number of clusters to use
    @affinity: metric used to calc linkage (default euclidean)
    @linkage: linkage criteria to use - determines which distance to use between sets of observation
        @ward: minimizes the variance of the clusters being merged
        @average: uses the average of the distances of each observation of the two sets
        @maximum: linkage uses the maximum distances between all observations of the two sets
        @single: uses the minimum of the distances between all observations of the two sets
    """

    # remove elements of list that show up less than k times
    def minify_list(lst, k):
        counted = Counter(lst)
        return [el for el in lst if counted[el] >= k]

    # find centroids in df w/ clusters
    def find_centroids(cl_df, clust):
        centroids, centroid_ids = [], []
        for c in set(clust):
            cl = cl_df.loc[cl_df['cluster_id'] == c]
            n, x, y = 0, 0, 0
            for idx, entry in cl.iterrows():
                x += entry['X']
                y += entry['Y']
                n += 1
            if n > 0:
                x /= n
                y /= n
                centroids.append((round(y, 3), round(x, 3)))
                centroid_ids.append(c)
        # print("generated centroids")
        return centroids, centroid_ids

    # cluster data
    def cluster(data, n_clust, d_threshold, min_size):
        # TODO: come up with a better way of handling this
        # translate string var props to their real value (unfortunately necessary because of text inputs)
        if n_clust != "None":
            d_threshold = None
            n_clust = int(n_clust)
        else:
            n_clust = None
            d_threshold = int(d_threshold)
        # actually run sklearn clustering function
        hc = AgglomerativeClustering(n_clusters=n_clust, distance_threshold=d_threshold, affinity=affinity, linkage=linkage)
        clust = hc.fit_predict(real_coords)
        # append cluster ids to df
        data['cluster_id'] = clust
        # setup random coords
        pb.update_progress(70)
        rand_coordinates = np.array(rand_coords)
        rand_cluster = hc.fit_predict(rand_coordinates)
        # fill random df
        rand_df = pd.DataFrame(rand_coordinates, columns=["X", "Y"])
        rand_df['cluster_id'] = rand_cluster
        # print("generated clusters")
        return data, rand_df, minify_list(clust, float(min_size)), minify_list(rand_cluster, float(min_size))

    def nnd(coordinate_list, random_coordinate_list):
        # finds nnd between centroids
        def distance_to_closest_particle(coord_list):
            nnd_list = []
            for z in range(len(coord_list)):
                small_dist = 10000000000000000000
                # og coord (x, y), closest coord (x, y), distance
                nnd_obj = [(0, 0), (0, 0), 0]
                p_if = (round(coord_list[z][1], 3), round(coord_list[z][0], 3))
                p_if_y, p_if_x = p_if
                nnd_obj[0] = p_if
                for j in range(0, len(coord_list)):
                    if z is not j:
                        p_jf = (round(coord_list[j][1], 3), round(coord_list[j][0], 3))
                        p_jf_y, p_jf_x = p_jf
                        dist = math.sqrt(((p_jf_y - p_if_y) ** 2) + ((p_jf_x - p_if_x) ** 2))
                        if dist < small_dist:
                            small_dist = round(dist, 3)
                            nnd_obj[1], nnd_obj[2] = p_jf, small_dist
                nnd_list.append(nnd_obj)
            # create new clean df
            clean_df = pd.DataFrame()
            if (len(nnd_list)) > 0:
                # clean and convert to df
                data = pd.DataFrame(data={'NND': nnd_list})
                # clean up df
                clean_df[['og_centroid', 'closest_centroid', 'dist']] = pd.DataFrame(
                    [x for x in data['NND'].tolist()])
                # print("generated nnd list")
            return clean_df
        # find nnd
        cleaned_real_df = distance_to_closest_particle(coordinate_list)
        cleaned_rand_df = distance_to_closest_particle(random_coordinate_list)
        # print("found nnd")
        return cleaned_real_df, cleaned_rand_df

    logging.info("running nearest neighbor distance between clusters")
    pb.update_progress(30)
    # cluster
    full_real_df, full_rand_df, cluster, rand_cluster = cluster(df, n_clusters, distance_threshold, min_clust_size)
    # generate centroids of clusters
    real_centroids, real_clust_ids = find_centroids(full_real_df, cluster)
    rand_centroids, rand_clust_ids = find_centroids(full_rand_df, rand_cluster)
    # run nearest neighbor distance on centroids
    real_df, rand_df = nnd(real_centroids, rand_centroids)
    # add back cluster ids to df
    real_df['cluster_id'] = real_clust_ids
    rand_df['cluster_id'] = rand_clust_ids
    # return dataframes with all elements, dataframe with only centroids and nnd
    print(real_df.head(), rand_df.head())
    return full_real_df, full_rand_df, real_df, rand_df


def draw_nnd_clust(nnd_df, clust_df, img, bin_counts, palette="rocket_r", circle_c=(0, 0, 255)):
    # color palette
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color
    # draw clusters
    cl_palette = create_color_pal(n_bins=len(set(clust_df['cluster_id'])), palette_type=palette)
    for idx, entry in clust_df.iterrows():
        particle = tuple(int(x) for x in [entry['X'], entry['Y']])
        img = cv2.circle(img, particle, 10, sea_to_rgb(cl_palette[int(clust_df['cluster_id'][idx])]), -1)
        # cv2.putText(img, str(clust_df['cluster_id'][idx]), org=particle, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=1)
    # draw nnd
    count, bin_idx = 0, 0
    for idx, entry in nnd_df.iterrows():
        count += 1
        particle_1 = tuple(int(x) for x in entry['og_centroid'])
        particle_2 = tuple(int(x) for x in entry['closest_centroid'])
        if count >= bin_counts[bin_idx] and bin_idx < len(bin_counts) - 1:
            bin_idx += 1
            count = 0
        img = cv2.circle(img, particle_1, 10, circle_c, -1)
        img = cv2.line(img, particle_1, particle_2, sea_to_rgb(palette[bin_idx]), 5)
        cv2.putText(img, str(int(clust_df['cluster_id'][idx])), org=particle_1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=1)
    return img


#
# def draw_nnd_clust(nnd_df, clust_df, img, bin_counts, palette="rocket_r", circle_c=(0, 0, 255)):
#     # color palette
#     def sea_to_rgb(color):
#         color = [val * 255 for val in color]
#         return color
#     # draw clusters
#     cl_palette = create_color_pal(n_bins=len(set(clust_df['cluster_id'])), palette_type=palette)
#     drawn_ids = []
#     for idx, entry in clust_df.iterrows():
#         particle = tuple(int(x) for x in [entry['X'], entry['Y']])
#         img = cv2.circle(img, particle, 10, sea_to_rgb(cl_palette[int(clust_df['cluster_id'][idx])]), -1)
#         c_id = str(clust_df['cluster_id'][idx])
#         if c_id not in drawn_ids:
#             cv2.putText(img, c_id, org=particle, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=1)
#             drawn_ids.append(c_id)
#     # draw nnd
#     count, bin_idx = 0, 0
#     for idx, entry in nnd_df.iterrows():
#         count += 1
#         particle_1 = tuple(int(x) for x in entry['og_centroid'])
#         particle_2 = tuple(int(x) for x in entry['closest_centroid'])
#         if count >= bin_counts[bin_idx] and bin_idx < len(bin_counts) - 1:
#             bin_idx += 1
#             count = 0
#         img = cv2.circle(img, particle_1, 10, circle_c, -1)
#         img = cv2.line(img, particle_1, particle_2, sea_to_rgb(palette[bin_idx]), 5)
#     return img