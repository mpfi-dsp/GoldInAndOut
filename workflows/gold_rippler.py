import logging

import pandas as pd
import numpy as np
import cv2

from typings import Unit
from utils import create_color_pal

COLORS = [(128, 0, 0),
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


def run_rippler(real_coords, rand_coords, spine_coords, img_path, mask_path, pb, max_steps=10, step_size=60):
    """
    GOLD RIPPLER (SC3PA)
    _______________________________
    @df: dataframe with centroids coordinates scaled to whatever format desired
    @pb: progress bar wrapper element, allows us to track how much time is left in process
    @rand_coords: list of randomly generated coordinates
    """
    logging.info("running gold rippler (SC3PA)")
    # print("running gold rippler (SC3PA)")
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

    # perm_scale_mask = np.zeros((img_size[0], img_size[1], 3), np.uint8)
    original_copy = img_og.copy()
    pb.update_progress(30)

    step = 0
    rippler_out = []
    for coord_list in [real_coords, rand_coords]:
        rad = 100
        max = (int(max_steps) * int(step_size)) + rad
        while rad <= max:
            total_captured_particles = 0
            scale_mask = np.zeros(pface_mask.shape, np.uint8)
            color_step = step % 11

            for entry in spine_coords:
                cv2.circle(scale_mask, entry, rad, 255, -1)
                cv2.circle(original_copy, entry, rad, COLORS[color_step], 5)

            step += 1
            for c in coord_list:
                x, y = int(c[1]), int(c[0])
                # cv2.circle(scale_mask, (y, x), rad, 255, -1)
                # cv2.circle(original_copy, (y, x), rad, COLORS[color_step], 5)
                if rad == max:
                    if scale_mask[y, x] != 0:
                        cv2.circle(original_copy, (y, x), 8, (0, 0, 255), -1)
                        total_captured_particles += 1
                    else:
                        cv2.circle(original_copy, (y, x), 8, (255, 0, 255), -1)
                else:
                    if scale_mask[y, x] != 0:
                        total_captured_particles += 1
                        cv2.circle(original_copy, (y, x), 8, (0, 0, 255), -1)
            gp_in_spine = total_captured_particles / len(coord_list)
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
            total_gp.append(len(coord_list))
            rad += int(step_size)
            pb.update_progress(rad)

        # generate new df and return
        new_df = pd.DataFrame(
            data={'radius': radius, '%_gp_captured': gp_captured, '%_pface_covered': pface_covered, 'SC3PA': SC3PA,
                  'total_gp': total_gp})
        # print(new_df.head())
        rippler_out.append(new_df)
        # cv2.imwrite("test_img.jpg", output_img)
    return rippler_out


def draw_rippler(coords, img, mask_path, palette="rocket_r", max_steps=10, step_size=60, scalar=1, input_unit=Unit.PIXEL, circle_c=(0, 0, 255)):
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color
    # print("draw rippler")
    output_img = img.copy()
    rad, step = 100, 0
    max = (int(max_steps) * int(step_size)) + rad
    pal = create_color_pal(n_bins=11, palette_type=palette)
    img_pface = cv2.imread(mask_path)
    # print("load imgs and find pface area")
    lower_bound = np.array([239, 174, 0])
    upper_bound = np.array([254, 254, 254])
    pface_mask = cv2.inRange(img_pface, lower_bound, upper_bound)
    while rad <= max:
        color_step = step % 11
        scale_mask = np.zeros(pface_mask.shape, np.uint8)
        for c in coords:
            x, y = int(c[1]), int(c[0])
            cv2.circle(scale_mask, (y, x), rad, 255, -1)
            cv2.circle(output_img, (y, x), rad, sea_to_rgb(pal[color_step]), 5)
            if rad == max:
                if scale_mask[y, x] != 0:
                    cv2.circle(output_img, (y, x), 8, (0, 0, 255), -1)
                else:
                    cv2.circle(output_img, (y, x), 8, (255, 0, 255), -1)
            else:
                if scale_mask[y, x] != 0:
                    cv2.circle(output_img, (y, x), 8, (0, 0, 255), -1)
        # for c in coords:
        #     x, y = int(c[1]), int(c[0])
        #     output_img = cv2.circle(output_img, (y, x), 8, circle_c, -1)
        #     output_img = cv2.circle(output_img, (y, x), rad, 255, -1)
        #     output_img = cv2.circle(output_img, (y, x), rad, sea_to_rgb(pal[color_step]), 5)
        rad += int(step_size)
        step += 1
    # print("done drawing")
    return output_img
