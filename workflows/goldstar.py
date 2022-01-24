import logging
import pandas as pd
import math
import cv2

def run_goldstar(real_coords, rand_coords, alt_coords, pb):
    """
    NEAREST NEIGHBOR DISTANCE
    _______________________________
    @real_coords: real coordinates scaled to whatever format desired
    @rand_coords: list of randomly generated coordinates
    @pb: progress bar wrapper element, allows us to track how much time is left in process
    """
    def goldstar_nnd(coordinate_list, random_coordinate_list, alt_coordinate_list):
        def goldstar_distance_closest(coord_list, alt_list):
            nnd_list = []
            for p in coord_list:
                pb.emit(p)
                small_dist = 10000000000000000000
                p1 = (p[1], p[0])
                nnd_obj = [p1, (0,0), 0]
                p_if_y, p_if_x = p1
                for j in alt_list:
                    if p != j:
                        p2 = (j[1], j[0])
                        p_jf_y, p_jf_x = p2
                        dist = math.sqrt(((p_jf_y - p_if_y) ** 2) + ((p_jf_x - p_if_x) ** 2))
                        if dist < small_dist:
                            small_dist = dist
                            nnd_obj[1], nnd_obj[2] = p2, small_dist
                nnd_list.append(nnd_obj)
            return nnd_list
        # def goldstar_distance_closest(coord_list, alt_list):
        #     nnd_list = []
        #     for z in range(len(coord_list)):
        #         pb.emit(z)
        #         d = 10000000000000000000
        #         p1 = (coord_list[z][1], coord_list[z][0])
        #         nnd_obj = [p1, (0, 0), 0]
        #         p_if_y, p_if_x = p1
        #         for j in range(0, len(alt_list)):
        #             if z is not j:
        #                 p2 = (coord_list[j][1], coord_list[j][0])
        #                 p_jf_y, p_jf_x = p2
        #                 dist = math.sqrt(((p_jf_y - p_if_y) ** 2) + ((p_jf_x - p_if_x) ** 2))
        #                 if dist < d:
        #                     d = dist
        #                     nnd_obj[1], nnd_obj[2] = p2, d
        #         nnd_list.append(nnd_obj)
        #     return nnd_list

        # find dist to closest particle goldstar
        logging.info("running goldstar nnd")
        real_goldstar_list = goldstar_distance_closest(coordinate_list, alt_coordinate_list)
        real_df = pd.DataFrame(data={'Nearest Neighbor Starfish Distance': real_goldstar_list})
        # clean up df
        clean_real_df = pd.DataFrame()
        clean_real_df[['og_coord', 'goldstar_coord', 'dist']] = pd.DataFrame(
            [x for x in real_df['Nearest Neighbor Starfish Distance'].tolist()])
        # find random dist
        random_goldstar_list = goldstar_distance_closest(random_coordinate_list, alt_coordinate_list)
        rand_df = pd.DataFrame(data={'Nearest Neighbor Starfish Distance': random_goldstar_list})
        # fill clean random df
        clean_rand_df = pd.DataFrame()
        clean_rand_df[['og_coord', 'goldstar_coord', 'dist']] = pd.DataFrame(
            [x for x in rand_df['Nearest Neighbor Starfish Distance'].tolist()])
        # print(clean_real_df.head())
        return clean_real_df, clean_rand_df

    # if generate_random prop enabled, create random coordinates and return results, else return real coordinates
    return goldstar_nnd(coordinate_list=real_coords, random_coordinate_list=rand_coords, alt_coordinate_list=alt_coords)


def draw_goldstar(nnd_df: pd.DataFrame, bin_counts: List[int], img: List, palette: List[Tuple[int, int, int]], circle_c: Tuple[int, int, int] = (0, 0, 255)):
    """ DRAW LINES TO ANNOTATE N NEAREST DIST ON IMAGE """
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color

    count, bin_idx = 0, 0
    for idx, entry in nnd_df.iterrows():
        count += 1
        particle_1 = tuple(int(x) for x in entry['og_coord'])
        particle_2 = tuple(int(x) for x in entry['goldstar_coord'])
        if count >= bin_counts[bin_idx] and bin_idx < len(bin_counts) - 1:
            bin_idx += 1
            count = 0
        img = cv2.circle(img, particle_1, 10, circle_c, -1)
        img = cv2.line(img, particle_1, particle_2, sea_to_rgb(palette[bin_idx]), 5)
        img = cv2.circle(img, particle_2, 10, (0, 0, 255), -1)

    for idx, entry in nnd_df.iterrows():
        particle_1 = tuple(int(x) for x in entry['og_coord'])
        cv2.putText(img, str(idx), org=particle_1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255),fontScale=0.5)
    return img
