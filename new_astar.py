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

def search(maze, cost, start, end, its = None):
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

    if its == None:
        max_iterations = (len(maze) // 2) ** 10
    else:
        max_iterations = its

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
            print("Iterations: " + str(outer_iterations))
            return return_path(current_node,maze)

        # Pop current node out off yet_to_visit list, add to visited list
        yet_to_visit_list.pop(current_index)
        visited_list.append(current_node)

        # test if goal is reached or not, if yes then return the path
        if current_node == end_node:
            return return_path(current_node,maze)
            print(outer_iterations)

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
            # print("New node added")

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

def holeSearch(maze, cost, start, end, its = None):
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

    if its == None:
        max_iterations = (len(maze) // 2) ** 10
    else:
        max_iterations = its

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
            print("Iterations: " + str(outer_iterations))
            return return_path(current_node,maze)

        # Pop current node out off yet_to_visit list, add to visited list
        yet_to_visit_list.pop(current_index)
        visited_list.append(current_node)

        # test if goal is reached or not, if yes then return the path
        if current_node == end_node:
            return return_path(current_node,maze)
            print(outer_iterations)

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
                return return_path(current_node,maze)
                # continue

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

def unique(list1):
  
    # initialize a null list
    unique_list = []
      
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    return unique_list

img_path = "./Feb 8 2022 P1 for test/P1 Bk6 Cav2_1 12nm montage.tif"
mask_path = "./Feb 8 2022 P1 for test/P1 Bk6 Cav2_1 12nm blue mask.tif"
# mask_path = "./Feb 8 2022 P1 for test/P2 Bk6 Cav2_1 12nm blue mask.tiff"
csv_path = "./Feb 8 2022 P1 for test/P1 XY 12nm in pixels.csv"
csv2_path = "./Feb 8 2022 P1 for test/P1 XY spines in pixels.csv"

data = pd.read_csv(csv_path, sep=",")
scaled_df = pixels_conversion(data=data, unit=Unit.PIXEL, scalar=1.0)
COORDS = to_coord_list(scaled_df)

data = pd.read_csv(csv2_path, sep=",")
ALT_COORDS = to_coord_list(
pixels_conversion(data=data, unit=Unit.PIXEL, scalar=1.0))

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
flip_grid = binary
# print(grid, grid.shape)
grid[grid == 255] = 1
flip_grid[flip_grid == 255] = 1
grid = grid^(grid&1==grid)
flip_grid = flip_grid^(flip_grid&1==flip_grid)

new_grid = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(grid)))
new_grid_flipped = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(flip_grid)))

for i, particle in enumerate(COORDS):
    COORDS[i] = (int(particle[0] * (1/8)), int(particle[1] * (1/8)))

for i, alt_coord in enumerate(ALT_COORDS):
    ALT_COORDS[i] = (int(alt_coord[0] * (1/8)), int(alt_coord[1] * (1/8)))

fig, ax = plt.subplots(figsize=(8,8))

start = tuple(int(x) for x in COORDS[2])
goal = tuple(int(x) for x in ALT_COORDS[2])

ax.scatter(start[1],start[0], marker = "*", color = "pink", s = 200)
ax.scatter(goal[1],goal[0], marker = "*", color = "grey", s = 200)

def makePath(maze, cost, start, end, its = None):
    path = search(maze, cost, start, end, its)

    uVals = []
    for _path in path:
        uVals.extend(unique(_path))

    uVals = list(filter(lambda val: val != -1, uVals))

    uArray =  np.asarray(path)
    xVals = []
    yVals = []

    for i in range(len(uArray)):
        for j in range(len(uArray[i])):
            if(uArray[i][j] in uVals):
                xVals.append(i)
                yVals.append(j)


    return xVals, yVals, uVals

def makeHolePath(maze, cost, start, end, its = None):
    path = holeSearch(maze, cost, start, end, its)

    last = path[-1]

    uVals = []
    for _path in path:
        uVals.extend(unique(_path))

    uVals = list(filter(lambda val: val != -1, uVals))

    uArray =  np.asarray(path)
    xVals = []
    yVals = []

    for i in range(len(uArray)):
        for j in range(len(uArray[i])):
            if(uArray[i][j] in uVals):
                xVals.append(i)
                yVals.append(j)


    return xVals, yVals, uVals

def astarMap(map, flippedMap, start, goal):
    print("Starting Hole Path...")
    _Xi, _Yi, _Ui = makeHolePath(flippedMap, 1, goal, start)
    print("Hole Path Finished")

    if((_Yi[-1], _Xi[-1]) == (goal[1],goal[0])):
        gPoint = (_Xi[1], _Yi[1])
    else:
        gPoint = (_Xi[-1], _Yi[-1])

    new_grid = cv2.circle(map, (gPoint[1],gPoint[0]), 3, (0, 0, 0), -1)

    print("Starting Paritcle Path...")
    _X, _Y, _U = makePath(map,1, start, gPoint)
    print("Particle Path Finished")

    ax.imshow(map, cmap=plt.cm.binary)
    plt.scatter(_Y, _X)
    plt.scatter(_Yi, _Xi)
    plt.show()

astarMap(new_grid, new_grid_flipped, start, goal)