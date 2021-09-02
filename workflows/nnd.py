from PyQt5.QtCore import pyqtSignal
from typing import List, Tuple
import logging
import pandas as pd
import math
import cv2


def run_nnd(real_coords: List[Tuple[float, float]], rand_coords: List[Tuple[float, float]], pb: pyqtSignal):
    """
    NEAREST NEIGHBOR DISTANCE
    _______________________________
    @real_coords: real coordinates scaled to whatever format desired
    @rand_coords: list of randomly generated coordinates
    @pb: progress bar wrapper element, allows us to track how much time is left in process
    """
    def nnd(coordinate_list: List[Tuple[float, float]], random_coordinate_list: List[Tuple[float, float]]):
        # find dist to closest particle
        def distance_to_closest_particle(coord_list):
            nnd_list = []
            for z in range(len(coord_list)):
                pb.emit(z)
                closest_d = 10000000000000000000
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
                        if dist < closest_d and dist != 0:
                            closest_d = dist
                            nnd_obj[1], nnd_obj[2] = p_jf, closest_d
                nnd_list.append(nnd_obj)
            return nnd_list

        logging.info("running nnd")
        real_nnd_list = distance_to_closest_particle(coordinate_list)
        real_df = pd.DataFrame(data={'Nearest Neighbor Distance': real_nnd_list})
        # clean up df
        clean_real_df = pd.DataFrame()
        clean_real_df[['og_coord', 'closest_coord', 'dist']] = pd.DataFrame(
            [x for x in real_df['Nearest Neighbor Distance'].tolist()])
        # find random dist
        random_nnd_list = distance_to_closest_particle(random_coordinate_list)
        rand_df = pd.DataFrame(data={'Nearest Neighbor Distance': random_nnd_list})
        # fill clean random df
        clean_rand_df = pd.DataFrame()
        clean_rand_df[['og_coord', 'closest_coord', 'dist']] = pd.DataFrame(
            [x for x in rand_df['Nearest Neighbor Distance'].tolist()])
        return clean_real_df, clean_rand_df

    return nnd(coordinate_list=real_coords, random_coordinate_list=rand_coords)


def draw_length(nnd_df: pd.DataFrame, bin_counts: List[int], img, palette, circle_c=(0, 0, 255)):
    """ DRAW LINES TO ANNOTATE N NEAREST DIST ON IMAGE """
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color

    count, bin_idx = 0, 0
    for idx, entry in nnd_df.iterrows():
        count += 1
        particle_1 = tuple(int(x) for x in entry['og_coord'])
        particle_2 = tuple(int(x) for x in entry['closest_coord'])
        if count >= bin_counts[bin_idx] and bin_idx < len(bin_counts) - 1:
            bin_idx += 1
            count = 0
        img = cv2.circle(img, particle_1, 10, circle_c, -1)
        img = cv2.line(img, particle_1, particle_2, sea_to_rgb(palette[bin_idx]), 5)

    for idx, entry in nnd_df.iterrows():
        particle_1 = tuple(int(x) for x in entry['og_coord'])
        cv2.putText(img, str(int(idx)), org=particle_1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=0.5)
    return img
