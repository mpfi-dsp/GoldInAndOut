import cv2
import numpy as np
import random
import pandas as pd
import math

def run_nnd(prog_wrapper, img_path, csv_path, pface_path="", csv_scalar=1, gen_rand=False):
    # Generate Faux Gold Particles within the P-face
    def generate_random_points(boundary, quantity, mask):
        coordinates = []
        count = 0
        while count <= quantity:
            x = random.randint(1, boundary[0] - 1)
            y = random.randint(1, boundary[1] - 1)
            if mask[x, y] != 0:
                coordinates.append((x, y))
                count += 1
        print(f"The total number of particles inside the pface are {count}.")
        return coordinates

    def distance_to_closest_particle(coord_list):
        nndlist = []
        coord_list_len = len(coord_list)
        # for progress bar

        for i in range(coord_list_len - 1):
            prog_wrapper.update_progress(i)
            small_dist = 10000000000000000000
            # og coord (x, y), closest coord (x, y), distance
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
                    if (dist < small_dist):
                        small_dist = dist
                        templist[1] = particle_jf
                        templist[2] = small_dist
            nndlist.append(templist)

        return nndlist

    def nnd(coordinate_list, pface_mask=[]):
        print("running nnd")
        real_nndlist = distance_to_closest_particle(coordinate_list)
        d = {'Nearest Neighbor Distance': real_nndlist}
        real_df = pd.DataFrame(data=d)

        if gen_rand and len(pface_mask) > 0:
            randomcoords = generate_random_points(pface_mask.shape, len(coordinate_list), pface_mask)
            random_nndlist = distance_to_closest_particle(randomcoords)
            d = {'Nearest Neighbor Distance': random_nndlist}
            rand_df = pd.DataFrame(data=d)
        # clean up df
        clean_df = pd.DataFrame()
        clean_df[['og_coord', 'closest_coord', 'dist']] = pd.DataFrame(
            [x for x in real_df['Nearest Neighbor Distance'].tolist()])
        return clean_df

    # Import Images
    img_original = cv2.imread(img_path)
    crop = img_original.shape

    # Import CSV Coordinates
    data = pd.read_csv(csv_path, sep=",", header=None)
    x_coordinates = np.array(data[1][1:])
    y_coordinates = np.array(data[2][1:])

    real_coordinates = []
    for i in range(len(x_coordinates)):
        x = round(float(x_coordinates[i]) * int(csv_scalar))
        y = round(float(y_coordinates[i]) * int(csv_scalar))
        real_coordinates.append((y, x))

    if gen_rand and len(pface_path) > 0:
        img_pface = cv2.imread(pface_path)
        img_pface = img_pface[:crop[0], :crop[1], :3]
        # Grab Contours of P-face
        lower_bound = np.array([239, 174, 0])
        upper_bound = np.array([254, 254, 254])
        pface_mask = cv2.inRange(img_pface, lower_bound, upper_bound)
        pface_cnts, pface_hierarchy = cv2.findContours(pface_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        drawn_pface_mask = img_pface.copy()
        drawn_pface_mask = cv2.drawContours(drawn_pface_mask, pface_cnts, -1, (0, 255, 0), 5)

        pface_area = 0
        for cnt in pface_cnts:
            area = cv2.contourArea(cnt)
            pface_area += area
        return nnd(real_coordinates, pface_mask)
    else:
        return nnd(real_coordinates)


def draw_length(nnd_df, img):
    for index, entry in nnd_df.iterrows():
        particle_1 = entry['og_coord']
        particle_2 = entry['closest_coord']
        print(particle_2)
        img = cv2.circle(img, particle_1, 10, (0, 0, 255), -1)
        img = cv2.line(img, particle_1, particle_2, (255, 255, 0), 5)

    # to save image:
    cv2.imwrite('drawn_nnd_img.jpg', img)
    return img
