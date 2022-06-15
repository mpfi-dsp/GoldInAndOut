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

class Node:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other):
        return self.position == other.position

#This function return the path of the search
def return_path(current_node,maze):
    path = []
    no_rows, no_columns = np.shape(maze)
    # here we create the initialized result maze with -1 in every position
    result = [[-1 for i in range(no_columns)] for j in range(no_rows)]
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    # Return reversed path as we need to show from start to end path
    path = path[::-1]
    start_value = 0
    # we update the path of start to end found by A-star serch with every step incremented by 1
    for i in range(len(path)):
        result[path[i][0]][path[i][1]] = start_value
        start_value += 1
    return result


def search(maze, cost, start, end):
    """
        Returns a list of tuples as a path from the given start to the given end in the given maze
        :param maze:
        :param cost
        :param start:
        :param end:
        :return:
    """

    # Create start and end node with initized values for g, h and f
    start_node = Node(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both yet_to_visit and visited list
    # in this list we will put all node that are yet_to_visit for exploration. 
    # From here we will find the lowest cost node to expand next
    yet_to_visit_list = []  
    # in this list we will put all node those already explored so that we don't explore it again
    visited_list = [] 
    
    # Add the start node
    yet_to_visit_list.append(start_node)
    
    # Adding a stop condition. This is to avoid any infinite loop and stop 
    # execution after some reasonable number of steps
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10

    # what squares do we search . serarch movement is left-right-top-bottom 
    #(4 movements) from every positon

    move  =  [[-1, 0 ], # go up
              [ 0, -1], # go left
              [ 1, 0 ], # go down
              [ 0, 1 ]] # go right


    """
        1) We first get the current node by comparing all f cost and selecting the lowest cost node for further expansion
        2) Check max iteration reached or not . Set a message and stop execution
        3) Remove the selected node from yet_to_visit list and add this node to visited list
        4) Perofmr Goal test and return the path else perform below steps
        5) For selected node find out all children (use move to find children)
            a) get the current postion for the selected node (this becomes parent node for the children)
            b) check if a valid position exist (boundary will make few nodes invalid)
            c) if any node is a wall then ignore that
            d) add to valid children node list for the selected parent
            
            For all the children node
                a) if child in visited list then ignore it and try next node
                b) calculate child node g, h and f values
                c) if child in yet_to_visit list then ignore it
                d) else move the child to yet_to_visit list
    """
    #find maze has got how many rows and columns 
    no_rows, no_columns = np.shape(maze)
    
    # Loop until you find the end
    
    while len(yet_to_visit_list) > 0:
        
        # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
        outer_iterations += 1    

        
        # Get the current node
        current_node = yet_to_visit_list[0]
        current_index = 0
        for index, item in enumerate(yet_to_visit_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
                
        # if we hit this point return the path such as it may be no solution or 
        # computation cost is too high
        if outer_iterations > max_iterations:
            print ("giving up on pathfinding too many iterations")
            return return_path(current_node,maze)

        # Pop current node out off yet_to_visit list, add to visited list
        yet_to_visit_list.pop(current_index)
        visited_list.append(current_node)

        # test if goal is reached or not, if yes then return the path
        if current_node == end_node:
            return return_path(current_node,maze)

        # Generate children from all adjacent squares
        children = []

        for new_position in move: 

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range (check if within maze boundary)
            if (node_position[0] > (no_rows - 1) or 
                node_position[0] < 0 or 
                node_position[1] > (no_columns -1) or 
                node_position[1] < 0):
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            
            # Child is on the visited list (search entire visited list)
            if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + cost
            ## Heuristic costs calculated here, this is using eucledian distance
            child.h = (((child.position[0] - end_node.position[0]) ** 2) + 
                       ((child.position[1] - end_node.position[1]) ** 2)) 

            child.f = child.g + child.h

            # Child is already in the yet_to_visit list and g cost is already lower
            if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                continue

            # Add the child to the yet_to_visit list
            yet_to_visit_list.append(child)


maze = [[0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0]]
    
start = [0, 0] # starting position
end = [4,5] # ending position
cost = 1 # cost per movement

path = search(maze,cost, start, end)

print('\n'.join([''.join(["{:" ">3d}".format(item) for item in row]) 
      for row in path]))

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

# def gaussian_pyramid(image, scale=2, minSize=(60, 60)):
#    yield image
#    while True:
#      w = int(image.shape[1] / scale)
#      h = int(image.shape[0] / scale)
#      image = cv2.pyrDown(image, dstsize=(w,h))
#      if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
#         break
#      yield image

# new_grid = gaussian_pyramid(grid, scale=2, minSize=(256, 256))
# new_grid = np.array(new_grid)
# print(new_grid)

# fig, ax = plt.subplots(figsize=(8,8))
# ax.imshow(grid, cmap=plt.cm.Dark2)
# ax.scatter(COORDS[0][1],COORDS[0][0], marker = "*", color = "yellow", s = 200)
# ax.scatter(ALT_COORDS[0][1],ALT_COORDS[0][0], marker = "*", color = "red", s = 200)
# plt.show()

new_grid = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(grid)))
print(new_grid, new_grid.shape)

for i, particle in enumerate(COORDS):
    COORDS[i] = (int(particle[0] * (1/8)), int(particle[1] * (1/8)))

for i, alt_coord in enumerate(ALT_COORDS):
    ALT_COORDS[i] = (int(alt_coord[0] * (1/8)), int(alt_coord[1] * (1/8)))

# run a star
for particle in COORDS:
    start = tuple(int(x) for x in particle)#[::-1]
    print('start', new_grid[start])
    if new_grid[start] == 0:
        for alt_coord in ALT_COORDS:
            goal = tuple(int(x) for x in alt_coord)#[::-1]
            print(start, goal)


            # plot map and path
            fig, ax = plt.subplots(figsize=(8,8))
            ax.imshow(new_grid, cmap=plt.cm.Dark2)
            ax.scatter(start[1],start[0], marker = "*", color = "yellow", s = 200)
            ax.scatter(goal[1],goal[0], marker = "*", color = "red", s = 200)
            plt.show()
            print('generating path')
            route = astar(new_grid, start, goal)
            route = route + [start]
            # Reverse the order:
            route = route[::-1]
            print(route)
            dist = len(route)
            small_dist = 1000000000000
            if dist < small_dist:
                small_dist = dist
                print('new smallest distance', dist)