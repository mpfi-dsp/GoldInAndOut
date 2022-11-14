import logging
import pandas as pd
from typing import List, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import pyqtSignal
from utils import pixels_conversion, enum_to_unit, to_coord_list
from typings import Unit, Workflow, DataObj, OutputOptions, WorkflowObj
from workflows.astar import run_astar, map_fill # Fill Workflow
# from workflows.astar import run_astar, map_fill # Fill Workflow
# from workflows.backupAstar import run_astar, map_fill # NoFill workflow
import cv2

# img_path = "./P4/P4 JCFFRIL25 Bk6 Cav2.1 12nm.tif"
# mask_path = "./P4/P4 wh bck blue mask.tif"
# csv_path = "./P4/P4 Results XY 12nm in pixels.csv"
# csv2_path = "./P4/P4 Spine Results XY in pixels.csv"

# img_path = "./P2/P2 Bk6 Cav2_1 12nm image.tif"
# mask_path = "./P2/P2 Bk6 Cav2_1 12nm blue mask.tif"
# csv_path = "./P2/P2 XY 12nm in pixels gold.csv"
# csv2_path = "./P2/P2 XY Spines in pixels landmark.csv"

# img_path = "./P3/P3 JCFFRIL25 Bk6 Cav2.1 12nm image.tif"
# mask_path = "./P3/P3 JCFFRIL25 Bk6 Cav2.1 12nm blue mask.tif"
# csv_path = "./P3/P3 XY 12nm in pixels gold.csv"
# csv2_path = "./P3/P3 XY Spines in pixels landmark.csv"

# data = pd.read_csv(csv_path, sep=",")
# scaled_df = pixels_conversion(data=data, unit=Unit.PIXEL, scalar=1.0)
# COORDS = to_coord_list(scaled_df)

# data = pd.read_csv(csv2_path, sep=",")
# ALT_COORDS = to_coord_list(
# pixels_conversion(data=data, unit=Unit.PIXEL, scalar=1.0))

# def run_goldAstar(map_path, mask_path, coord_list: List[Tuple[float, float]], alt_list: List[Tuple[float, float]]):
def run_goldAstar(map_path, mask_path, coord_list: List[Tuple[float, float]], alt_list: List[Tuple[float, float]], pb: pyqtSignal):
    def dist2(p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

    def fuse(points, d):
        ret = []
        d2 = d * d
        n = len(points)
        taken = [False] * n
        for i in range(n):
            if not taken[i]:
                count = 1
                point = [points[i][0], points[i][1]]
                taken[i] = True
                for j in range(i+1, n):
                    if dist2(points[i], points[j]) < d2:
                        point[0] += points[j][0]
                        point[1] += points[j][1]
                        count+=1
                        taken[j] = True
                point[0] /= count
                point[1] /= count
                ret.append((point[0], point[1]))
        return ret

    def nearestPointOnLine(pt, r0, r1, clipToSegment = True):
        r01 = r1 - r0           # vector from r0 to r1 
        d = np.linalg.norm(r01) # length of r01
        r01u = r01 / d          # unit vector from r0 to r1
        r = pt - r0             # vector from r0 to pt
        rid = np.dot(r, r01u)   # projection (length) of r onto r01u
        ri = r01u * rid         # projection vector
        lpt = r0 + ri           # point on line

        if clipToSegment:       # if projection is not on line segment
            if rid > d:         # clip to endpoints if clipToSegment set
                return r1
            if rid < 0:
                return r0 

        return lpt

    def goldstar_distance_closest(coord_list: List[Tuple[float, float]], alt_list: List[Tuple[float, float]]):
        nnd_list = []
        for p in coord_list:
            small_dist = 10000000000000000000
            p1 = (p[1], p[0])
            nnd_obj = [p1, (0,0), 0]
            p_if_y, p_if_x = p1
            for j in alt_list:
                p2 = (j[1], j[0])
                if p1 != p2:
                    p_jf_y, p_jf_x = p2
                    dist = math.sqrt(((p_jf_y - p_if_y) ** 2) + ((p_jf_x - p_if_x) ** 2))
                    if dist < small_dist:
                        small_dist = dist
                        nnd_obj[1], nnd_obj[2] = p2, small_dist
            nnd_list.append(nnd_obj)
        return nnd_list

    nnd_df = pd.DataFrame()
    nnd_df[['og_coord', 'goldstar_coord', 'dist']] = pd.DataFrame(goldstar_distance_closest(coord_list, alt_list))

    mask = cv2.imread(mask_path)                                        # Load imported mask
    img = cv2.imread(map_path)                                          # Load important image
    cMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)                      # Convery mask to Black & White
    ret, binary = cv2.threshold(cMask, 220, 255, cv2.THRESH_BINARY)     # Threshold on mask to make it binary
    base = np.zeros_like(mask)                                          # Create blank image with mask dimensions
    imgMaskMerge = cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)            # create image merging img & mask

    binary = map_fill(binary)

    for idx, entry in nnd_df.iterrows():
        particle_1 = tuple(int(x) for x in entry['og_coord'])
        particle_2 = tuple(int(x) for x in entry['goldstar_coord'])

        base = cv2.circle(base, particle_1, 10, (0, 0, 255), -1)        # Draw particle on base
        base = cv2.line(base, particle_1, particle_2, (0, 0, 255), 5)   # Draw particle -> landmark line on base
        base = cv2.circle(base, particle_2, 10, (0, 0, 255), -1)        # Draw landmark on base

    maskedLineImg = cv2.bitwise_or(base, base, mask = binary)           # Mask out base with mask (who could've guessed)
    maskedLineImg = cv2.cvtColor(maskedLineImg, cv2.COLOR_BGR2GRAY)     # Convert new masked image to Black & White

    # Get contours of  masked image
    contours, hierarchy = cv2.findContours(maskedLineImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cList = []
    for c in contours:

        # Compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cList.append((cX, cY))  # Add center to list of all centers

    mergedCList = fuse(cList, 200)  # If centers are close to each other...
                                    # delete them and add an average of their points instead

    newParticleList = nnd_df.copy()

    astarList = []
    nonSelected_Point = []
    nonSelected_Landmark = []
    nonSelected_Distance = []

    for p in mergedCList:   
        dists = [math.sqrt((p[0]-s1)**2 + (p[1]-s0)**2) for s0, s1 in alt_list]

        if(all(i >= 100 for i in dists)):
            for idx, entry in newParticleList.iterrows():
                particle_1 = tuple(int(x) for x in entry['og_coord'])
                p1_array = np.asarray(particle_1)
                particle_2 = tuple(int(x) for x in entry['goldstar_coord'])
                dist_ = entry['dist']
                p2_array = np.asarray(particle_2)

                imgMaskMerge = cv2.circle(imgMaskMerge, p1_array, 10, (255, 0, 0), -1)
                imgMaskMerge = cv2.circle(imgMaskMerge, p2_array, 10, (255, 0, 0), -1)

                pt = nearestPointOnLine(p, p1_array, p2_array, True)
                dist = np.linalg.norm(p-pt)

                if(dist >= 100):
                    imgMaskMerge = cv2.line(imgMaskMerge, p1_array, p2_array, (255, 0, 0), 5)
                else:
                    imgMaskMerge = cv2.line(imgMaskMerge, p1_array, p2_array, (0, 0, 255), 5)               

                    astarList.append(particle_1)
                    newParticleList.drop(idx, inplace=True)

    for idx, entry in newParticleList.iterrows():
        particle_1 = tuple(int(x) for x in entry['og_coord'])
        particle_2 = tuple(int(x) for x in entry['goldstar_coord'])
        dist = entry['dist']

        nonSelected_Point.append(particle_1)
        nonSelected_Landmark.append(particle_2)
        nonSelected_Distance.append(dist)

    print("A* Length: {}".format(len(astarList)))

    # astarDF, astarDF_, astarCoords, astarCoords_ = run_astar(map_path, mask_path, astarList[84:], alt_list)
    astarDF, astarDF_, astarCoords, astarCoords_ = run_astar(map_path, mask_path, astarList, alt_list, pb)
    
    nonSelected_DF = pd.DataFrame(list(zip(nonSelected_Point, nonSelected_Landmark, nonSelected_Distance,
                np.zeros(len(nonSelected_Point)), np.zeros(len(nonSelected_Point)), nonSelected_Distance)),
                columns =['og_coord', 'astar_coord', 'goldstar_dist', 'astar_dist', 'smoothed_dist', 'dist'])

    combined_astarDF = pd.concat([astarDF, nonSelected_DF])

    astarCoords = astarCoords.set_index('og_coord')
    astarCoords = astarCoords.reindex(index=astarDF['og_coord'])
    astarCoords = astarCoords.reset_index()

    return combined_astarDF, combined_astarDF, astarCoords, astarCoords
    # return astarDF, astarDF, astarCoords, astarCoords

# run_goldAstar(img_path, mask_path, COORDS, ALT_COORDS)

def draw_goldAstar(nnd_df: pd.DataFrame, path_df: pd.DataFrame, bin_counts: List[int], img: List, mask: List, palette: List[Tuple[int, int, int]], alt_palette: List[Tuple[int, int, int]], circle_c: Tuple[int, int, int] = (0, 0, 255)):
    
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(mask, 100, 255, cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    out = np.zeros_like(binary)
    contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, 255, cv2.FILLED)

    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    img = cv2.addWeighted(img, 0.5, out, 0.5, 0)
    
    """ DRAW LINES TO ANNOTATE N NEAREST DIST ON IMAGE """
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color

    count, bin_idx, pth_idx = 0, 0, 0
    for idx, entry in nnd_df.iterrows():
        count += 1
        particle_1 = tuple(int(x) for x in entry['og_coord'])
        particle_2 = tuple(int(x) for x in entry['astar_coord'])
        if count >= bin_counts[bin_idx] and bin_idx < len(bin_counts) - 1:
            bin_idx += 1
            count = 0

        if(entry['astar_dist'] == 0):
            img = cv2.line(img, particle_1, particle_2, sea_to_rgb(palette[bin_idx]), 5)
        else:
            paths = path_df['Path']
            current_path = np.array([paths[pth_idx]])
            img = cv2.polylines(img, np.int32([current_path]), False, sea_to_rgb(alt_palette[bin_idx]), 5)
            pth_idx += 1
        
        img = cv2.circle(img, particle_1, 10, circle_c, -1)
        img = cv2.circle(img, particle_2, 10, (0, 0, 255), -1)

    for idx, entry in nnd_df.iterrows():
        particle_1 = tuple(int(x) for x in entry['og_coord'])
        cv2.putText(img, str(idx), org=particle_1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=0.5)
    
    return img