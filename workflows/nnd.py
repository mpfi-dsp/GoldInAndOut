import pandas as pd
import numpy as np
import math
import cv2
from typings import Unit

def run_nnd(df, pb, rand_coords):
    """
    NEAREST NEIGHBOR DISTANCE
    _______________________________
    @df: dataframe with coordinates scaled to whatever format desired
    @pb: progress bar wrapper element, allows us to track how much time is left in process
    @rand_coords: list of randomly generated coordinates
    """
    def nnd(coordinate_list, random_coordinate_list):
        # find dist to closest particle
        def distance_to_closest_particle(coord_list):
            nnd_list = []
            for z in range(len(coord_list)):
                pb.update_progress(z)
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
            return nnd_list

        print("running nnd")
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

    # FIND NND
    x_coordinates = np.array(df['X'])
    y_coordinates = np.array(df['Y'])
    real_coordinates = []
    for i in range(len(x_coordinates)):
        real_coordinates.append((float(y_coordinates[i]), float(x_coordinates[i])))
    # if generate_random prop enabled, create random coordinates and return results, else return real coordinates
    return nnd(coordinate_list=real_coordinates, random_coordinate_list=rand_coords)


def draw_length(nnd_df, bin_counts, img, palette, input_unit=Unit.PIXEL, scalar=1, circle_c=(0, 0, 255)):
    """ DRAW LINES TO ANNOTATE N NEAREST DIST ON IMAGE """
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color

    count, bin_idx = 0, 0
    for idx, entry in nnd_df.iterrows():
        count += 1
        particle_1 = entry['og_coord']
        particle_2 = entry['closest_coord']
        if input_unit == Unit.PIXEL:
            particle_1 = tuple(int(scalar * x) for x in particle_1)
            particle_2 = tuple(int(scalar * x) for x in particle_2)
        else:
            particle_1 = tuple(int(x / scalar) for x in particle_1)
            particle_2 = tuple(int(x / scalar) for x in particle_2)
        if count >= bin_counts[bin_idx] and bin_idx < len(bin_counts) - 1:
            bin_idx += 1
            count = 0
        img = cv2.circle(img, particle_1, 10, circle_c, -1)
        img = cv2.line(img, particle_1, particle_2, sea_to_rgb(palette[bin_idx]), 5)
    return img
