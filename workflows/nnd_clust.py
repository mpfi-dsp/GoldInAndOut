import math

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import cv2
from utils import create_color_pal
from collections import Counter

"""
NEAREST NEIGHBOR DISTANCE OF WARD HIERARCHICAL CLUSTERING
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


def run_nnd_clust(df, prog, random_coordinate_list, min_clust_size=3, distance_threshold=120, n_clusters=None,
                  affinity='euclidean', linkage='ward'):
    def minify_list(lst, k):
        counted = Counter(lst)
        return [el for el in lst if counted[el] >= k]

    def find_centroids(cl_df, clust):
        centroids = []
        centroid_ids = []
        # print(set(clust))
        for c in set(clust):
            # print(c)
            cl = cl_df.loc[cl_df['cluster_id'] == c]
            # print(cl_df.loc[cl_df['cluster_id'] == c])
            n = 0
            x = 0
            y = 0
            for idx, entry in cl.iterrows():
                x += entry['X']
                y += entry['Y']
                n += 1
            if n > 0:
                x /= n
                y /= n
                centroids.append((round(y, 3), round(x, 3)))
                centroid_ids.append(c)
        # print(centroids)
        print("generated centroids")
        return centroids, centroid_ids

    def cluster(data, n_clust, d_threshold, min_size):
        # print(distance_threshold, n_clusters)
        if n_clust != "None":
            d_threshold = None
            n_clust = int(n_clust)
        else:
            n_clust = None
            d_threshold = int(d_threshold)
        hc = AgglomerativeClustering(n_clusters=n_clust, distance_threshold=d_threshold, affinity=affinity,
                                     linkage=linkage)
        clust = hc.fit_predict(real_coordinates)
        data['cluster_id'] = clust

        # random coords
        prog.update_progress(70)
        rand_coordinates = np.array(random_coordinate_list)
        rand_cluster = hc.fit_predict(rand_coordinates)

        rand_df = pd.DataFrame(rand_coordinates, columns=["X", "Y"])
        rand_df['cluster_id'] = rand_cluster

        # for _df in [data, rand_df]:
        #     _id = 0
        #     count = 0
        #     for i in _df['cluster_id']:
        #         if i == _id:
        #             count += 1
        #         else:
        #             if count < float(min_size):
        #                 _df = _df[_df['cluster_id'] != _id]
        #
        #             _id = i
        #             count = 0

        print("generated clusters")
        return data, rand_df, minify_list(clust, float(min_size)), minify_list(rand_cluster, float(min_size))

    def nnd(coordinate_list, random_coordinate_list):
        """ FIND DIST TO CLOSEST PARTICLE """

        def distance_to_closest_particle(coord_list):
            nndlist = []
            # print(coord_list)
            # coord_list_len = len(coord_list)
            # # print(coord_list_len)
            # for i in range(coord_list_len - 1):
            for i in range(len(coord_list)):
                # update progress bar
                small_dist = 10000000000000000000
                # templist = [og coord (x, y), closest coord (x, y), distance]
                templist = [0, 0, 0]
                particle_i = coord_list[i]
                particle_if = (round(particle_i[1], 3), round(particle_i[0], 3))
                templist[0] = particle_if
                for j in range(0, len(coord_list)):
                    if i is not j:
                        particle_j = coord_list[j]
                        particle_jf = (round(particle_j[1], 3), round(particle_j[0], 3))
                        dist = ((particle_jf[0] - particle_if[0]) ** 2) + ((particle_jf[1] - particle_if[1]) ** 2)
                        dist = math.sqrt(dist)
                        if dist < small_dist:
                            small_dist = round(dist, 3)
                            templist[1] = particle_jf
                            templist[2] = small_dist
                nndlist.append(templist)

            clean_df = pd.DataFrame()
            if (len(nndlist)) > 0:
                # clean and convert to df
                d = {'NND': nndlist}
                data = pd.DataFrame(data=d)
                # clean up df
                clean_df[['og_centroid', 'closest_centroid', 'dist']] = pd.DataFrame(
                    [x for x in data['NND'].tolist()])
                print("generated nnd list")
            return clean_df

        cleaned_real_df = distance_to_closest_particle(coordinate_list)
        cleaned_rand_df = distance_to_closest_particle(random_coordinate_list)

        print("found nnd")
        return cleaned_real_df, cleaned_rand_df

    print("nearest neighbor distance between clusters")
    # turn into coordinate list
    x_coordinates = np.array(df['X'])
    y_coordinates = np.array(df['Y'])
    real_coordinates = []
    for i in range(len(x_coordinates)):
        real_coordinates.append([float(y_coordinates[i]), float(x_coordinates[i])])

    # make numpy array
    prog.update_progress(30)
    real_coordinates = np.array(real_coordinates)

    # cluster
    real_df, rand_df, cluster, rand_cluster = cluster(df, n_clusters, distance_threshold, min_clust_size)

    # generate centroids of clusters
    real_centroids, real_clust_ids = find_centroids(real_df, cluster)
    rand_centroids, rand_clust_ids = find_centroids(rand_df, rand_cluster)

    # print(real_centroids, rand_centroids)
    clean_real_df, clean_rand_df = nnd(real_centroids, rand_centroids)
    # print(clean_real_df.head())
    print(clean_real_df, real_clust_ids)
    clean_real_df['cluster_id'] = real_clust_ids
    clean_rand_df['cluster_id'] = rand_clust_ids
    print(clean_real_df.head())

    return real_df, rand_df, clean_real_df, clean_rand_df


def draw_nnd_clust(cluster_df, img, palette="rocket_r", scalar=1):
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color

    palette = create_color_pal(n_bins=len(set(cluster_df['cluster_id'])), palette_type=palette)

    for idx, entry in cluster_df.iterrows():
        particle = tuple(int(scalar * x) for x in [entry['X'], entry['Y']])
        img = cv2.circle(img, particle, 10, sea_to_rgb(palette[cluster_df['cluster_id'][idx]]), -1)
    return img
