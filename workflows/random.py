import cv2
import numpy as np
import random

def gen_random_coordinates(data, img_path="", mask_path="", count=None):
    """
    RANDOM COORDS GENERATOR
    _______________________________
    @data: dataframe with coordinates scaled to whatever format desired
    @img_path: path to image we are finding the n nearest distance of (only needed if gen_rand is True)
    @pface_path: path to mask we are finding the n nearest distance of (only needed if gen_rand is True)
    @n_rand_to_gen: number of random particles to generate
    """
    def generate_random_points(boundary, quantity, mask):
        # generate faux particles within the pface
        coords = []
        count = 0
        while count <= quantity:
            x = random.randint(1, boundary[0] - 1)
            y = random.randint(1, boundary[1] - 1)
            if mask[x, y] != 0:
                coords.append((x, y))
                count += 1
        # print(f"The total number of particles inside the p-face are {count}.")
        return coords

    x_coordinates = np.array(data['X'])
    y_coordinates = np.array(data['Y'])
    read_coords = []
    for i in range(len(x_coordinates)):
        read_coords.append((float(y_coordinates[i]), float(x_coordinates[i])))

    # import img
    img_original = cv2.imread(img_path)
    crop = img_original.shape
    img_pface = cv2.imread(mask_path)
    img_pface = img_pface[:crop[0], :crop[1], :3]
    # grab contours of pface
    lower_bound = np.array([239, 174, 0])
    upper_bound = np.array([254, 254, 254])
    pface_mask = cv2.inRange(img_pface, lower_bound, upper_bound)
    # pface_cnts, pface_hierarchy = cv2.findContours(pface_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # pface_area = 0
    #
    # for cnt in pface_cnts:
    #     area = cv2.contourArea(cnt)
    #     pface_area += area

    return generate_random_points(pface_mask.shape, count, pface_mask)
