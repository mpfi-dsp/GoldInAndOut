import pandas as pd
import numpy as np
import cv2

from typings import Unit
from utils import create_color_pal


def run_rippler(df, img_path, mask_path, pb, rand_coords, max_steps=10, step_size=60):
    """
    GOLD RIPPLER (SC3PA)
    _______________________________
    @df: dataframe with centroids coordinates scaled to whatever format desired
    @pb: progress bar wrapper element, allows us to track how much time is left in process
    @rand_coords: list of randomly generated coordinates
    """
    print("running gold rippler (SC3PA)")
    # find Spine Correlated Particles Per P-face Area (SC3PA)
    img_og = cv2.imread(img_path)
    img_pface = cv2.imread(mask_path)
    # print("load imgs and find pface area")
    lower_bound = np.array([239, 174, 0])
    upper_bound = np.array([254, 254, 254])
    pface_mask = cv2.inRange(img_pface, lower_bound, upper_bound)
    # find pface area
    pface_cnts, pface_hierarchy = cv2.findContours(pface_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    pface_cnts2, pface_hierarchy2 = cv2.findContours(pface_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    pface_area_external, pface_area_tree = 0, 0
    for cnt in pface_cnts2:
        area = cv2.contourArea(cnt)
        pface_area_external += area
    for cnt in pface_cnts:
        area = cv2.contourArea(cnt)
        pface_area_tree += area

    difference = (pface_area_tree - pface_area_external)
    pface_area = pface_area_external - difference

    SC3PA, radius, gp_captured, pface_covered, total_gp = [], [], [], [], []
    step = 0

    color_list = [(128, 0, 0),
                  (139, 0, 0),
                  (165, 42, 42),
                  (178, 34, 34),
                  (220, 20, 60),
                  (255, 0, 0),
                  (255, 99, 61),
                  (255, 127, 80),
                  (205, 92, 92),
                  (240, 128, 128),
                  (233, 150, 122)]
    # perm_scale_mask = np.zeros((img_size[0], img_size[1], 3), np.uint8)
    original_copy = img_og.copy()
    output_img = img_og.copy()

    # turn into coordinate list
    x_coordinates = np.array(df['X'])
    y_coordinates = np.array(df['Y'])
    coords = []
    for i in range(len(x_coordinates)):
        coords.append([float(y_coordinates[i]), float(x_coordinates[i])])

    rippler_out = []
    for coord_list in [coords, rand_coords]:
        rad = 100
        max = (int(max_steps) * int(step_size)) + rad
        while rad <= max:
            total_captured_particles = 0
            scale_mask = np.zeros(pface_mask.shape, np.uint8)
            color_step = step % 11
            for entry in coord_list:
                output_img = cv2.circle(scale_mask, tuple(int(x) for x in entry), rad, 255, -1)
                output_img = cv2.circle(original_copy, tuple(int(x) for x in entry), rad, color_list[color_step], 5)
            step += 1
            for c in coord_list:
                x, y = int(c[1]), int(c[0])
                if rad == max:
                    if scale_mask[y, x] != 0:
                        output_img = cv2.circle(original_copy, (y, x), 8, (0, 0, 255), -1)
                        total_captured_particles += 1
                    else:
                        output_img = cv2.circle(original_copy, (y, x), 8, (255, 0, 255), -1)
                else:
                    if scale_mask[y, x] != 0:
                        total_captured_particles += 1
                        output_img = cv2.circle(original_copy, (y, x), 8, (0, 0, 255), -1)
            gp_in_spine = total_captured_particles / len(df)
            # find spine contour area and pface contour area
            mask_combined = cv2.bitwise_and(scale_mask, pface_mask.copy())
            mask_cnts, mask_hierarchy = cv2.findContours(mask_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            mask_cnts2, mask_hierarchy2 = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # find scale area
            scale_area_external = 0
            scale_area_tree = 0
            # find cts
            for cnt in mask_cnts2:
                area = cv2.contourArea(cnt)
                scale_area_external += area
            for cnt in mask_cnts:
                area = cv2.contourArea(cnt)
                scale_area_tree += area
            # find stats
            difference = (scale_area_tree - scale_area_external)
            scale_area = scale_area_external - difference
            percent_area = scale_area / pface_area
            # calculate SC3PA
            scaled_SC3PA = gp_in_spine / percent_area
            SC3PA.append(scaled_SC3PA)
            radius.append(rad)
            gp_captured.append(gp_in_spine)
            pface_covered.append(percent_area)
            total_gp.append(len(df))
            rad += int(step_size)
            pb.update_progress(rad)

        # generate new df and return
        new_df = pd.DataFrame(
            data={'radius': radius, '%_gp_captured': gp_captured, '%_pface_covered': pface_covered, 'SC3PA': SC3PA,
                  'total_gp': total_gp})
        print(new_df.head())
        rippler_out.append(new_df)

        cv2.imwrite("test_img.jpg", output_img)

    return rippler_out


def draw_rippler(nnd_df, clust_df, img, bin_counts, palette="rocket_r", input_unit=Unit.PIXEL, scalar=1,
                 circle_c=(0, 0, 255)):
    return None
#     # color palette
#     def sea_to_rgb(color):
#         color = [val * 255 for val in color]
#         return color
#     # draw clusters
#     cl_palette = create_color_pal(n_bins=len(set(clust_df['cluster_id'])), palette_type=palette)
#     for idx, entry in clust_df.iterrows():
#         particle = tuple(int(scalar * x) for x in [entry['X'], entry['Y']])
#         img = cv2.circle(img, particle, 10, sea_to_rgb(cl_palette[clust_df['cluster_id'][idx]]), -1)
#     # draw nnd
#     count, bin_idx = 0, 0
#     for idx, entry in nnd_df.iterrows():
#         count += 1
#         particle_1 = entry['og_centroid']
#         particle_2 = entry['closest_centroid']
#         if input_unit == Unit.PIXEL:
#             particle_1 = tuple(int(scalar * x) for x in particle_1)
#             particle_2 = tuple(int(scalar * x) for x in particle_2)
#         else:
#             particle_1 = tuple(int(x / scalar) for x in particle_1)
#             particle_2 = tuple(int(x / scalar) for x in particle_2)
#         if count >= bin_counts[bin_idx] and bin_idx < len(bin_counts) - 1:
#             bin_idx += 1
#             count = 0
#         img = cv2.circle(img, particle_1, 10, circle_c, -1)
#         img = cv2.line(img, particle_1, particle_2, sea_to_rgb(palette[bin_idx]), 5)
#     return img
