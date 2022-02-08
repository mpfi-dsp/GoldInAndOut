import logging
import numpy as np
import random
import cv2
from globals import DEFAULT_DISTANCE_THRESH 

def gen_random_coordinates(img_path: str, mask_path: str, count: int = 0):
    """
    RANDOM COORDS GENERATOR
    _______________________________
    @img_path: path to image
    @mask_path: path to mask
    @count: number of random particles to generate
    """
    def generate_random_points(boundary: list, quantity: int, mask: list):
        # generate pseudo-random distribution of particles within the p-face
        def meets_threshold(new_point, points):
            for point in points:
                dist = np.sqrt(np.sum(np.square(new_point-point)))
                if dist < DEFAULT_DISTANCE_THRESH:
                    return False
            return True

        def generate_K_points(K):
            points = []
            while len(points) < K:
                x = random.randint(1, boundary[0] - 1)
                y = random.randint(1, boundary[1] - 1)
                if mask[x, y] != 0:
                    new_point = np.array([x, y])
                    if meets_threshold(new_point, points):
                        points.append(new_point)
            return points

        return generate_K_points(quantity)

    if len(img_path) == 0:
        return []
    # import img
    img_original = cv2.imread(img_path)
    crop = img_original.shape
    # if no mask provided, use the entire image
    if len(mask_path) > 0:
        img_pface = cv2.imread(mask_path)
    else:
        img_pface = np.zeros(crop, dtype=np.uint8)
        img_pface.fill(245)
    # crop to size of normal image
    img_pface = img_pface[:crop[0], :crop[1], :3]
    # convert to grayscale
    img_pface2 = cv2.cvtColor(img_pface, cv2.COLOR_BGR2GRAY)
    # # convert to binary
    ret, binary = cv2.threshold(img_pface2, 100, 255, cv2.THRESH_OTSU)
    pface_mask = ~binary
    # Alternative method of grabbing contours of pface
    # lower_bound = np.array([239, 174, 0])
    # upper_bound = np.array([254, 254, 254])
    # pface_mask: list = cv2.inRange(img_pface, lower_bound, upper_bound)
    # print(pface_mask.shape, pface_mask)
    logging.info("Generated random particles")
    return generate_random_points(pface_mask.shape, count, pface_mask)
