import math

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import cv2
from utils import create_color_pal

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
def run_nnd_clust(df, prog, random_coordinate_list, min_clust_size=3, distance_threshold=120, n_clusters=None, affinity='euclidean', linkage='ward'):

    def find_centroids(cl_df, clust):
        centroids = []
        for j in range(len(set(clust))):
            cl = cl_df.loc[cl_df['cluster_id'] == j]
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
                centroids.append((y, x))
        # print(centroids)
        print("generated centroids")
        return centroids

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

        # g = data.groupby(['cluster_id'])
        # minified_data = g.filter(lambda x: len(x) >= 3)
        # print(minified_data)

        # random coords
        prog.update_progress(70)
        rand_coordinates = np.array(random_coordinate_list)
        rand_cluster = hc.fit_predict(rand_coordinates)

        rand_df = pd.DataFrame(rand_coordinates, columns=["X", "Y"])
        rand_df['cluster_id'] = rand_cluster
        print("generated clusters")
        return data, rand_df, clust, rand_cluster

    def nnd(coordinate_list, random_coordinate_list):
        """ FIND DIST TO CLOSEST PARTICLE """
        def distance_to_closest_particle(coord_list):
            nndlist = []
            # print(coord_list)
            coord_list_len = len(coord_list)
            # print(coord_list_len)
            for i in range(coord_list_len - 1):
                # update progress bar
                small_dist = 10000000000000000000
                # templist = [og coord (x, y), closest coord (x, y), distance]
                templist = [0, 0, 0]
                particle_i = coord_list[i]
                particle_if = (particle_i[1], particle_i[0])
                templist[0] = particle_if
                for j in range(0, coord_list_len - 1):
                    if i is not j:
                        particle_j = coord_list[j]
                        particle_jf = (particle_j[1], particle_j[0])
                        dist = ((particle_jf[0] - particle_if[0]) ** 2) + ((particle_jf[1] - particle_if[1]) ** 2)
                        dist = math.sqrt(dist)
                        if dist < small_dist:
                            small_dist = dist
                            templist[1] = particle_jf
                            templist[2] = small_dist
                nndlist.append(templist)

            return nndlist

        # print("mask", pface_mask, pface_mask.shape)
        real_nndlist = distance_to_closest_particle(coordinate_list)
        print(real_nndlist)

        d = {'NND': real_nndlist}
        real_df = pd.DataFrame(data=d)
        # clean up df
        clean_real_df = pd.DataFrame()
        clean_real_df[['og_centroid', 'closest_centroid', 'dist']] = pd.DataFrame(
            [x for x in real_df['NND'].tolist()])

        random_nndlist = distance_to_closest_particle(random_coordinate_list)
        d = {'NND': random_nndlist}
        rand_df = pd.DataFrame(data=d)

        clean_rand_df = pd.DataFrame()
        clean_rand_df[['og_centroid', 'closest_centroid', 'dist']] = pd.DataFrame(
            [x for x in rand_df['NND'].tolist()])

        return clean_real_df, clean_rand_df


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
    real_centroids = find_centroids(real_df, cluster)
    rand_centroids = find_centroids(rand_df, rand_cluster)

    clean_real_df, clean_rand_df = nnd(real_centroids, rand_centroids)

    print(clean_real_df)

    clean_real_df['cluster_id'] = real_df['cluster_id']
    clean_rand_df['cluster_id'] = rand_df['cluster_id']

    print(clean_real_df.head())

    return clean_real_df, clean_rand_df


def draw_clust(cluster_df, img, workflow, palette="rocket_r", scalar=1):
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color

    palette = create_color_pal(n_bins=len(set(cluster_df[workflow['graph']['x_type']])), palette_type=palette)

    for idx, entry in cluster_df.iterrows():
        particle = tuple(int(scalar * x) for x in [entry['X'], entry['Y']])
        img = cv2.circle(img, particle, 10, sea_to_rgb(palette[cluster_df[workflow['graph']['x_type']][idx]]), -1)
    return img
