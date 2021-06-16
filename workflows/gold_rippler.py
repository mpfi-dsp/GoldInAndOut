import pandas as pd
import numpy as np
import cv2

def run_rippler(df, img_size, pface_mask, pface_area, img_orig, maxsteps, stepsize):
    SC3PA = []
    Radius = []
    GPCaptured = []
    PfaceCovered = []
    TotalGP = []
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
    original_copy = img_orig.copy()

    """ NND BETWEEN CLUSTERS """
    print("running nearest neighbor distance between clusters")
    # turn into coordinate list
    x_coordinates = np.array(df['X'])
    y_coordinates = np.array(df['Y'])
    real_coordinates = []
    for i in range(len(x_coordinates)):
        real_coordinates.append([float(y_coordinates[i]), float(x_coordinates[i])])

    circleRadius = 100
    max = (maxsteps * stepsize) + circleRadius

    while circleRadius <= max:

        total_captured_particles = 0
        scale_mask = np.zeros(img_size, np.uint8)
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
        Radius.append(circleRadius)
        GPCaptured.append(GP_within_spine)
        PfaceCovered.append(percent_area)
        TotalGP.append(len(df))
        circleRadius += stepsize

    d = {'Radius': Radius, 'Percent GP Captured': GPCaptured, 'Percent P-face Covered': PfaceCovered, 'SC3PA': SC3PA,
         'Total GP': TotalGP}
    df = pd.DataFrame(data=d)

    print(df.head())

    return df, original_copy
