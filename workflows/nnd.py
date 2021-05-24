import pandas as pd
import numpy as np
import math
import cv2
from typings import Unit

""" 
NND RUN FUNCTION
_______________________________
@data: dataframe with coordinates scaled to whatever format desired
@prog: progress bar wrapper element, allows us to track how much time is left in process
@random_coordinate_list: list of randomly generated coordinates
"""
def run_nnd(data, prog, random_coordinate_list):
    """ NEAREST NEIGHBOR DISTANCE """
    def nnd(coordinate_list, random_coordinate_list):
        """ FIND DIST TO CLOSEST PARTICLE """
        def distance_to_closest_particle(coord_list):
            nndlist = []
            coord_list_len = len(coord_list)
            for i in range(coord_list_len - 1):
                # update progress bar
                prog.update_progress(i)
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
        print("running nnd")
        real_nndlist = distance_to_closest_particle(coordinate_list)
        d = {'Nearest Neighbor Distance': real_nndlist}
        real_df = pd.DataFrame(data=d)
        # clean up df
        clean_real_df = pd.DataFrame()
        clean_real_df[['og_coord', 'closest_coord', 'dist']] = pd.DataFrame(
            [x for x in real_df['Nearest Neighbor Distance'].tolist()])

        random_nndlist = distance_to_closest_particle(random_coordinate_list)
        d = {'Nearest Neighbor Distance': random_nndlist}
        rand_df = pd.DataFrame(data=d)

        clean_rand_df = pd.DataFrame()
        clean_rand_df[['og_coord', 'closest_coord', 'dist']] = pd.DataFrame(
            [x for x in rand_df['Nearest Neighbor Distance'].tolist()])

        return clean_real_df, clean_rand_df

    """ START """
    x_coordinates = np.array(data['X'])
    y_coordinates = np.array(data['Y'])
    real_coordinates = []
    for i in range(len(x_coordinates)):
        real_coordinates.append((float(y_coordinates[i]), float(x_coordinates[i])))

    # if generate_random prop enabled, create random coordinates and return results, else return real coordinates
    return nnd(coordinate_list=real_coordinates, random_coordinate_list=random_coordinate_list)


""" DRAW LINES TO ANNOTATE N NEAREST DIST ON IMAGE """
def draw_length(nnd_df, bin_counts, img, palette, input_unit=Unit.PIXEL, scalar=1, save_img=False, circle_c=(0, 0, 255)):
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color
    count = 0
    bin_idx = 0
    print(nnd_df.head())
    for idx, entry in nnd_df.iterrows():
        count += 1
        # print(idx, count)
        particle_1 = entry['og_coord']
        particle_2 = entry['closest_coord']
        # scale back to drawable size
        # print("test", input_unit == 'px')
        if input_unit == Unit.PIXEL:
            particle_1 = tuple(int(scalar * x) for x in particle_1)
            particle_2 = tuple(int(scalar * x) for x in particle_2)
        else:
            particle_1 = tuple(int(x / scalar) for x in particle_1)
            particle_2 = tuple(int(x / scalar) for x in particle_2)
        if count >= bin_counts[bin_idx] and bin_idx < len(bin_counts) - 1:
            bin_idx += 1
            count = 0
        # print(particle_1)
        img = cv2.circle(img, particle_1, 10, circle_c, -1)
        img = cv2.line(img, particle_1, particle_2, sea_to_rgb(palette[bin_idx]), 5)
    # save image
    if save_img:
        cv2.imwrite('../output/drawn_nnd_img.jpg', img)
    return img
