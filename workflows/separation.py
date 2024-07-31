import logging
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from globals import REAL_COLOR
from utils import create_color_pal, to_df
from collections import Counter
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from typing import List, Tuple
import numpy as np
import math
import cv2
from workflows import random_coords


def run_separation(pb: pyqtSignal, real_coords: List[Tuple[float, float]], rand_coords: List[Tuple[float, float]], min_clust_size: int = 2, distance_threshold: int = 27, affinity: str = 'euclidean', linkage: str = 'single', clust_area: bool = False):
    """
    NEAREST NEIGHBOR DISTANCE OF HIERARCHICAL CLUSTERING
    _______________________________
    @pb: progress bar wrapper element, allows us to track how much time is left in process
    @real_coords: list of real coordinates
    @rand_coords: list of randomly generated coordinates
    @min_clust_size: minimum number of coords required to be considered a "cluster"
    @distance_threshold: using a distance threshold to automatically cluster particles
    @affinity: metric used to calc linkage (default euclidean)
    @linkage: linkage criteria to use - determines which distance to use between sets of observation
        @single: uses the minimum of the distances between all observations of the two sets
        @ward: minimizes the variance of the clusters being merged
        @average: uses the average of the distances of each observation of the two sets
        @maximum: linkage uses the maximum distances between all observations of the two sets
    """

    # remove elements of list that show up fewer than k times
    def minify_list(lst: List[int], k: int = 2):
        counted = Counter(lst)
        return [el for el in lst if counted[el] >= k]

    # find centroids in df w/ clusters
    def find_centroids(cl_df: pd.DataFrame, clust: List[int]):
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
                centroids.append((y, x))
                centroid_ids.append(c)
        print("generated centroids")
        return centroids, centroid_ids

    # cluster data
    def cluster(coords: List[Tuple[float, float]], r_coords: List[Tuple[float, float]], d_threshold: int = 27, min_size: int = 2):
        n_clust = None
        # actually run sklearn clustering function
        hc = AgglomerativeClustering(n_clusters=n_clust, distance_threshold=d_threshold * 2, affinity=affinity,
                                     linkage=linkage)
        clust = hc.fit_predict(coords)
        # append cluster ids to df
        df = to_df(coords)
        df['cluster_id'] = clust
        # setup random coords
        rand_coordinates = np.array(r_coords)
        if random_coords.N > 1:
            len_real = len(real_coords)
            rand_coordinates_chunk = [rand_coordinates[i:i+len_real] for i in range(0, len(rand_coords), len_real)]
            rand_cluster = [x for rand_coordinates_chunk in [hc.fit_predict(x) for x in rand_coordinates_chunk] for x in rand_coordinates_chunk]
            blank_list = rand_cluster[0:len_real] # pick out same number of rand particles as real input
            first_trial = rand_cluster[0:len_real] # same as 'blank'
            double_len_real = len_real * 2
            # ensuring cluster ids are unique between trials
            for i in range(random_coords.N - 1):
                max_id = int(max(first_trial)) + 1
                testing = [x+max_id for x in rand_cluster[len_real:double_len_real]]
                first_trial += testing
                blank_list += testing
                len_real += len(real_coords)
                double_len_real += len(real_coords)
            rand_cluster = np.array(blank_list)
        else:
            rand_cluster = hc.fit_predict(rand_coordinates)
        rand_coordinates = np.flip(rand_coordinates, 1)
        pb.emit(70)
        # fill random df
        rand_df = pd.DataFrame(rand_coordinates, columns=["X", "Y"])
        rand_df['cluster_id'] = rand_cluster

        return df, rand_df, minify_list(clust, min_size), minify_list(rand_cluster, min_size)

    def nnd(coordinate_list: List[Tuple[float, float]], random_coordinate_list: List[Tuple[float, float]]):
        # finds nnd between centroids. Same as in nnd.py, but repeated here in case anything needs to be tweaked specifically for this workflow. GIO design stipulates that each workflow should be self-sufficient and contain all the code it needs to run.
        def distance_to_closest_particle(coord_list):
            nnd_list = []
            for z in range(len(coord_list)):
                small_dist = 10000000000000000000
                # og coord (x, y), closest coord (x, y), distance
                nnd_obj = [(0, 0), (0, 0), 0]
                p_if = (coord_list[z][1], coord_list[z][0])
                p_if_y, p_if_x = p_if
                nnd_obj[0] = p_if
                for j in range(0, len(coord_list)):
                    p_jf = (coord_list[j][1], coord_list[j][0])
                    if z is not j and p_if is not p_jf:
                        p_jf_y, p_jf_x = p_jf
                        dist = math.sqrt(((p_jf_y - p_if_y) ** 2) + ((p_jf_x - p_if_x) ** 2))
                        if dist < small_dist and dist != 0:
                            small_dist = dist
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
            return clean_df
        # find nnd
        cleaned_real_df = distance_to_closest_particle(coordinate_list)
        cleaned_rand_df = distance_to_closest_particle(random_coordinate_list)
        return cleaned_real_df, cleaned_rand_df

    def separation_correlation(trial_number):
        p = 0
        loop_counter = 1
        M = 0
        random_separation_df = pd.DataFrame()
        final_avg_rand = pd.DataFrame()
        updated_rand_centroids = rand_centroids.copy()
        for i in range(trial_number):
            reference_df_loop = reference_df.copy()
            sorted_IDs = full_rand_df.sort_values(by="cluster_id")
            sorted_IDs = sorted_IDs.reset_index()
            # within one trial of coordinates
            sorted_IDs_2 = sorted_IDs[sorted_IDs.iloc[:, 0] < # <=
                            int((len(full_rand_df) // random_coords.N) * loop_counter)]
            # more trials = increase the lower threshold
            if M > 0:
                sorted_IDs_2 = sorted_IDs_2[sorted_IDs_2.iloc[:, 0] >= M]
            ID_count = sorted_IDs_2.iloc[:, 3].value_counts()
            # find cluster IDs that show up more than the min clust size criteria (input)
            sorted_IDs_2 = sorted_IDs_2[sorted_IDs_2.iloc[:, 3].isin(ID_count.index[ID_count.gt(min_clust_size - 1)])]
            if sorted_IDs_2.empty: # no clusters within that trial
                sorted_IDs_2 = pd.DataFrame(np.array([[0, 0, 0, -1]]), columns=['index', 'X', 'Y', 'cluster_id'])
            # find values with an index less than the coord length -- within one random trial of the full df
            # find the maximum cluster ID within the trial
            max_ID = int(max(sorted_IDs_2.iloc[:, 3]))
            # find the position of the maximum index
            sorted = sorted_IDs_2.loc[sorted_IDs_2.iloc[:, 3] == max_ID]
            sorted = sorted.iloc[0, 3]
            # extract just coordinate (no ID or index) of maximum cluster ID
            if -1 in reference_df.iloc[:, 1]:
                reference_df_2 = pd.DataFrame(np.array([[(0, 0), 0]]), columns=['og_rand_centroid', 'correlating_ID'])
            else:
                reference_df_2 = reference_df.loc[reference_df_loop.iloc[:, 1] == sorted]
            reference_df_2 = reference_df_2.iloc[0, 0] # tuple
            # finding coordinate to correlate with centroid list; convert to int
            [int(x) for x in list(reference_df_2)] # https://stackoverflow.com/questions/7368789/convert-all-strings-in-a-list-to-integers
            # find the index of the coordinate in the centroid list
            if reference_df_2 == (0, 0): # if there is no cluster
                thresh = updated_rand_centroids.index(reference_df_2) + 1 # exclusive of actual thresh number without + 1
            else:
                thresh = rand_centroids.index(reference_df_2) + 1 # exclusive of actual thresh number without + 1
            # find nnd within centroid area
            if updated_rand_centroids[p:int(thresh)] == []:
                rand_df = pd.DataFrame(np.array([[0, 0, 0]]), columns=['og_centroid', 'closest_centroid', 'dist'])
            else:
                real_unused, rand_df = nnd(real_centroids, updated_rand_centroids[p:int(thresh)]) # goes to max trial point
            random_separation_df = pd.concat([random_separation_df, rand_df], ignore_index=True)
            distance = rand_df.iloc[:, 2].copy()
            insert_rows = int(thresh - p - 1)
            avg = pd.DataFrame(distance)
            avg = pd.DataFrame([avg.iloc[:, 0].mean()])
            avg = pd.concat([avg,
                pd.DataFrame(0, columns=avg.iloc[1:, 0], index=range(insert_rows))], ignore_index=True)
            avg = avg.fillna(0)
            final_avg_rand = pd.concat([final_avg_rand, avg], ignore_index=True)
            if loop_counter < random_coords.N:
                p = int(thresh)
                loop_counter += 1
                M += int(len(full_rand_df) // random_coords.N)
            if reference_df_2 == (0, 0):
                updated_rand_centroids[(thresh - 1)] = 'q' # signifies that there was a value of (0, 0) here and should not be indexed in the following loops
        rand_df = random_separation_df
        return rand_df, final_avg_rand

    logging.info("running nearest neighbor distance between clusters")
    pb.emit(30)
    # cluster
    full_real_df, full_rand_df, cluster, rand_cluster = cluster(real_coords, rand_coords, distance_threshold, min_clust_size)
    # generate centroids of clusters
    real_centroids, real_clust_ids = find_centroids(full_real_df, cluster)
    if random_coords.N > 1:
        len_real_2 = len(real_coords) # chunk row size
        # size of cluster ID df chunks
        rand_clust_list = full_rand_df['cluster_id'].tolist()
        rand_clust_chunks = [rand_clust_list[i:i+len_real_2] for i in range(0, len(full_rand_df), len_real_2)]
        minify_index = 0
        for i in range(random_coords.N): # if there are no repeated values, should just be the length of proto rand clust chunj
            rand_clust_chunks[minify_index] = minify_list(rand_clust_chunks[minify_index], min_clust_size)
            rand_clust_chunks[minify_index] = list(dict.fromkeys(rand_clust_chunks[minify_index]))
            if len(rand_clust_chunks[minify_index]) == 0: # if there are no clusters in that "chunk" equal to/above the min clust threshold
                rand_clust_chunks[minify_index] = [-1]
            rand_clust_chunks[minify_index].sort() # ensuring the values are in ascending order
            minify_index += 1
        random_chunks = [full_rand_df[i:i+len_real_2] for i in range(0, len(full_rand_df), len_real_2)]
        rand_centroid_list = []
        rand_clust_list = []
        centroid_index = 0
        minify_index_2 = 0
        for i in range(random_coords.N):
            if -1 not in rand_clust_chunks[minify_index_2]:
                    rand_centroids, rand_clust_ids = find_centroids(random_chunks[centroid_index], rand_clust_chunks[centroid_index])
                    rand_clust_ids.sort()
                    # https://stackoverflow.com/questions/6618515/sorting-list-according-to-corresponding-values-from-a-parallel-list
                    rand_centroids = [x for _, x in sorted(zip(rand_clust_ids, rand_centroids))] # need to sort properly for later on in separation correlation 
                    rand_centroid_list += rand_centroids
                    rand_clust_list += rand_clust_ids
                    if centroid_index < (random_coords.N - 1):
                        centroid_index += 1
            else:
                rand_centroid_list += [(0, 0)]
                rand_clust_list += [-1]
                if centroid_index < (random_coords.N - 1):
                    centroid_index += 1
            minify_index_2 += 1
        rand_centroids = []
        for i in rand_centroid_list:
            if i == (0,0) or i not in rand_centroids:
                rand_centroids.append(i) # https://stackoverflow.com/questions/7944895/how-to-remove-all-duplicates-from-list-except-one-element-in-python
        rand_clust_ids = []
        for i in rand_clust_list:
            if i == -1 or i not in rand_clust_ids:
                rand_clust_ids.append(i)
    else:
        rand_centroids, rand_clust_ids = find_centroids(full_rand_df, rand_cluster)
    # run nearest neighbor distance on centroids
    if random_coords.N > 1:
        # defining real df
        real_df, rand_unused = nnd(real_centroids, rand_centroids[0:len(real_centroids)])
        reference_df = pd.DataFrame()
        reference_df['og_rand_centroid'] = rand_centroids.copy()
        reference_df['correlating_ID'] = rand_clust_ids.copy()
        rand_centroids = reference_df['og_rand_centroid'].tolist() # in the same order for later loc index
        if -1 in rand_centroids and random_coords.N == 2:
            correlation_output = separation_correlation(trial_number=1)
            rand_df = correlation_output[0]
            final_avg_rand = correlation_output[1]
        elif -1 in rand_centroids and random_coords.N > 2 or -1 not in rand_centroids:
            correlation_output = separation_correlation(trial_number=random_coords.N)
            rand_df = correlation_output[0]
            final_avg_rand = correlation_output[1]
        final_avg_real = pd.DataFrame()
        distance_real = rand_df.iloc[:, 2].copy()
        insert_rows = len(real_df) - 1
        avg_real = pd.DataFrame(distance_real)
        avg_real = pd.DataFrame([avg_real.iloc[:, 0].mean()])
        avg_real = pd.concat([avg_real,
            pd.DataFrame(0, columns=avg_real.iloc[1:, 0], index=range(insert_rows))], ignore_index=True)
        avg_real = avg_real.fillna(0)
        final_avg_real = pd.concat([final_avg_real, avg_real], ignore_index=True)
    else:
        real_df, rand_df = nnd(real_centroids, rand_centroids)
    # add back cluster ids to df
    real_df['cluster_id'] = real_clust_ids
    rand_df['cluster_id'] = rand_clust_ids
    if rand_df.empty:
        rand_df = pd.DataFrame(np.array([[0, 0, 0, 0, 0]]), columns=['og_centroid', 'closest_centroid', 'dist', 'cluster_id', 'avg_dist'])
    if random_coords.N > 1:
        rand_df['avg_trial_dist'] = final_avg_rand.iloc[:, 0].values
    else:
        distance = rand_df.iloc[:, 2].copy()
        insert_rows = len(rand_df) - 1
        avg = pd.DataFrame(distance)
        avg = pd.DataFrame([avg.iloc[:, 0].mean()])
        avg = pd.concat([avg,
            pd.DataFrame(0, columns=avg.iloc[1:, 0], index=range(insert_rows))], ignore_index=True)
        avg = avg.fillna(0)
        rand_df['avg_trial_dist'] = avg

    # average distance
    len_full_real_df = len(full_real_df)
    # detailed random
    if random_coords.N > 1:
        for i in range(random_coords.N):
            if len(full_rand_df[0:]) > len_full_real_df:
                bottom_rows = full_rand_df.loc[len_full_real_df:, :]
                bottom_rows = bottom_rows.reset_index(drop=True)
                A = len_full_real_df * random_coords.N
                N_minus_one = random_coords.N - 1
                B = len_full_real_df * N_minus_one
                to_delete = A - B
                full_rand_df = full_rand_df.head(-to_delete)
                full_rand_df = pd.merge(full_rand_df, bottom_rows, how='outer', left_index=True, right_index=True)
                full_rand_df = full_rand_df.loc[:, ~full_rand_df.apply(lambda x: x.duplicated(), axis=1).all()].copy()
    # return dataframes with all elements, dataframe with only centroids and nnd
    return full_real_df, full_rand_df, real_df, rand_df


def draw_separation(nnd_df: pd.DataFrame, clust_df: pd.DataFrame, img: List, bin_counts: List[int], palette: List[Tuple[int, int, int]], circle_c: Tuple[int, int, int] = (0, 0, 255), distance_threshold: int = 27, draw_clust_area: bool = False, clust_area_color: Tuple[int, int, int] = REAL_COLOR):
    # color palette
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color

    if draw_clust_area:
        new_img = np.zeros(img.shape, dtype=np.uint8)
        new_img.fill(255)
    # draw clusters
    cl_palette = create_color_pal(n_bins=len(set(clust_df['cluster_id'])), palette_type=palette)
    for idx, entry in clust_df.iterrows(): # [:(len(clust_df)) // random_coords.N]
        particle = tuple(int(x) for x in [entry['X'], entry['Y']])
        img = cv2.circle(img, particle, 10, sea_to_rgb(cl_palette[int(clust_df['cluster_id'][idx])]), -1)
        if draw_clust_area:
            new_img = cv2.circle(new_img, particle, radius=distance_threshold, color=(0, 255, 0), thickness=-1)
    # draw nnd
    if draw_clust_area:
        lower_bound = np.array([0, 250, 0])
        upper_bound = np.array([40, 255, 40])
        clust_mask = cv2.inRange(new_img, lower_bound, upper_bound)
        clust_cnts, clust_hierarchy = cv2.findContours(clust_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        img = cv2.drawContours(img, clust_cnts, -1, clust_area_color, 3)

    count, bin_idx = 0, 0
    if len(nnd_df) > 1: # prevents one cluster and the origin from being labeled (line goes off graph)
        for idx, entry in nnd_df.iterrows():
            count += 1
            particle_1 = tuple(int(x) for x in entry['og_centroid'])
            particle_2 = tuple(int(x) for x in entry['closest_centroid'])
            if count >= bin_counts[bin_idx] and bin_idx < len(bin_counts) - 1:
                bin_idx += 1
                count = 0
            img = cv2.circle(img, particle_1, 10, circle_c, -1)
            img = cv2.line(img, particle_1, particle_2, sea_to_rgb(palette[bin_idx]), 5)
            cv2.putText(img, str(int(nnd_df['cluster_id'][idx])), org=particle_1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=1)
            # TODO: if you desire centroid area
            # if draw_clust_area:
            #     img = cv2.circle(img, particle_1, radius=int(distance_threshold), color=(0, 255, 0))
    return img
