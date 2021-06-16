import pandas as pd
import numpy as np
import cv2


def run_rippler(df, img_path, mask_path, max_steps=10, step_size=60):
    img_og = cv2.imread(img_path)
    img_pface = cv2.imread(mask_path)

    lower_bound = np.array([239, 174, 0])
    upper_bound = np.array([254, 254, 254])
    pface_mask = cv2.inRange(img_pface, lower_bound, upper_bound)
    # find pface area
    pface_cnts, pface_hierarchy = cv2.findContours(pface_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    pface_cnts2, pface_hierarchy2 = cv2.findContours(pface_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pface_area_external = 0
    pface_area_tree = 0
    for cnt in pface_cnts2:
        area = cv2.contourArea(cnt)
        pface_area_external += area
    for cnt in pface_cnts:
        area = cv2.contourArea(cnt)
        pface_area_tree += area
    difference = (pface_area_tree - pface_area_external)
    pface_area = pface_area_external - difference

    SC3PA = []
    radius = []
    gp_captured = []
    pface_covered = []
    total_gp = []
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

    # find Spine Correlated Particles Per P-face Area (SC3PA)
    print("running gold rippler (SC3PA)")
    # turn into coordinate list
    x_coordinates = np.array(df['X'])
    y_coordinates = np.array(df['Y'])
    real_coordinates = []
    for i in range(len(x_coordinates)):
        real_coordinates.append([float(y_coordinates[i]), float(x_coordinates[i])])

    circleRadius = 100
    max = (max_steps * step_size) + circleRadius

    while circleRadius <= max:

        total_captured_particles = 0
        scale_mask = np.zeros(pface_mask.shape, np.uint8)
        colorstep = step % 11

        for entry in real_coordinates:
            cv2.circle(scale_mask, entry, circleRadius, 255, -1)
            cv2.circle(original_copy, entry, circleRadius, color_list[colorstep], 5)

        step += 1

        # Find Number of Gold Particles Within Spine Contours
        # cv2_imshow(scale_mask)
        for pair in df:
            if (circleRadius == max):
                if scale_mask[pair[0], pair[1]] != 0:
                    total_captured_particles += 1
                    cv2.circle(original_copy, (pair[1], pair[0]), 8, (0, 0, 255), -1)
                else:
                    cv2.circle(original_copy, (pair[1], pair[0]), 8, (255, 0, 255), -1)
            else:
                if scale_mask[pair[0], pair[1]] != 0:
                    total_captured_particles += 1
                    cv2.circle(original_copy, (pair[1], pair[0]), 8, (0, 0, 255), -1)

        GP_within_spine = total_captured_particles / len(df)

        # Find Spine Contour Area / P-face Contour Area
        mask_combined = cv2.bitwise_and(scale_mask, pface_mask.copy())
        mask_cnts, mask_hierarchy = cv2.findContours(mask_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        mask_cnts2, mask_hierarchy2 = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # find scale area
        scale_area_external = 0
        scale_area_tree = 0

        for cnt in mask_cnts2:
            area = cv2.contourArea(cnt)
            scale_area_external += area

        for cnt in mask_cnts:
            area = cv2.contourArea(cnt)
            scale_area_tree += area

        difference = (scale_area_tree - scale_area_external)
        scale_area = scale_area_external - difference

        percent_area = scale_area / pface_area

        # Calculate SC3PA

        scaled_SC3PA = GP_within_spine / percent_area
        SC3PA.append(scaled_SC3PA)
        radius.append(circleRadius)
        gp_captured.append(GP_within_spine)
        pface_covered.append(percent_area)
        total_gp.append(len(df))
        circleRadius += step_size

    d = {'Radius': radius, 'Percent GP Captured': gp_captured, 'Percent P-face Covered': pface_covered, 'SC3PA': SC3PA,
         'Total GP': total_gp}
    new_df = pd.DataFrame(data=d)

    print(new_df.head())

    return new_df, original_copy
