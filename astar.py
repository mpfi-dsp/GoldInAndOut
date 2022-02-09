# now do it again but with real data
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import cv2
from utils import pixels_conversion, enum_to_unit, to_coord_list
import pandas as pd
from typings import Unit, Workflow, DataObj, OutputOptions, WorkflowObj

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def astar(array, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()

    came_from = {}

    gscore = {start:0}

    fscore = {start:heuristic(start, goal)}

    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
 

    while oheap:

        current = heapq.heappop(oheap)[1]

        if current == goal:

            data = []

            while current in came_from:

                data.append(current)

                current = came_from[current]

            return data
        close_set.add(current)

        for i, j in neighbors:

            neighbor = current[0] + i, current[1] + j            

            tentative_g_score = gscore[current] + heuristic(current, neighbor)

            if 0 <= neighbor[0] < array.shape[0]:

                if 0 <= neighbor[1] < array.shape[1]:                

                    if array[neighbor[0]][neighbor[1]] == 1:

                        continue

                else:

                    # array bound y walls

                    continue

            else:

                # array bound x walls

                continue
 

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):

                continue
 

            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:

                came_from[neighbor] = current

                gscore[neighbor] = tentative_g_score

                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                # print(oheap, (fscore[neighbor], neighbor))
                heapq.heappush(oheap, (fscore[neighbor], neighbor)) 

img_path = "./Feb 8 2022 P1 for test/P1 Bk6 Cav2_1 12nm montage.tif"
mask_path = "./Feb 8 2022 P1 for test/P1 Bk6 Cav2_1 12nm blue mask.tif"
csv_path = "./Feb 8 2022 P1 for test/P1 XY 12nm in pixels.csv"
csv2_path = "./Feb 8 2022 P1 for test/P1 XY spines in pixels.csv"

data = pd.read_csv(csv_path, sep=",")
scaled_df = pixels_conversion(data=data, unit=Unit.PIXEL, scalar=1.0)
COORDS = to_coord_list(scaled_df)

data = pd.read_csv(csv2_path, sep=",")
ALT_COORDS = to_coord_list(
pixels_conversion(data=data, unit=Unit.PIXEL, scalar=1.0))

# data = pd.read_csv(csv_path, sep=",")
# scaled_df = pixels_conversion(data=data, unit=Unit.NANOMETER, scalar=0.00112486)
# COORDS = to_coord_list(scaled_df)

# data = pd.read_csv(csv2_path, sep=",")
# ALT_COORDS = to_coord_list(
# pixels_conversion(data=data, unit=Unit.NANOMETER, scalar=0.00112486))

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
grid = ~binary
# print(grid, grid.shape)
grid[grid == 255] = 1
grid = grid^(grid&1==grid)
# grid[255] = 0
# grid = np.where((grid==0)|(grid==1), grid^1, grid)
# grid = ~grid + 
print(grid, grid.shape)
 
# run a star
for particle in COORDS:
    start = tuple(int(x) for x in particle)#[::-1]
    print('start', grid[start])
    if grid[start] == 0:
        for alt_coord in ALT_COORDS:
            # start = tuple((grid.shape[1] - int(particle[1]), grid.shape[0] - int(particle[0])))#[::-1]
            # goal = tuple((grid.shape[1] - int(alt_coord[1]), grid.shape[0] - int(alt_coord[0])))
            goal = tuple(int(x) for x in alt_coord)#[::-1]
            print(start, goal)


            # plot map and path
            # fig, ax = plt.subplots(figsize=(8,8))
            # ax.imshow(grid, cmap=plt.cm.Dark2)
            # ax.scatter(start[1],start[0], marker = "*", color = "yellow", s = 200)
            # ax.scatter(goal[1],goal[0], marker = "*", color = "red", s = 200)
            # plt.show()
            print('generating path')
            route = astar(grid, start, goal)
            route = route + [start]
            # Reverse the order:
            route = route[::-1]
            print(route)
            dist = len(route)
            small_dist = 1000000000000
            if dist < small_dist:
                small_dist = dist
                print('new smallest distance', dist)

# x_coords = []

# y_coords = []
# for i in (range(0,len(route))):

#     x = route[i][0]

#     y = route[i][1]

#     x_coords.append(x)

#     y_coords.append(y)
# plot map and path

# fig, ax = plt.subplots(figsize=(6,7))

# ax.imshow(grid, cmap=plt.cm.Dark2)

# ax.scatter(start[1],start[0], marker = "*", color = "yellow", s = 200)

# ax.scatter(goal[1],goal[0], marker = "*", color = "red", s = 200)

# ax.plot(y_coords,x_coords, color = "black")

# plt.show()
 
# print('distance', len(route))