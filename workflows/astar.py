# Abandon all hope ye who enter beyond this point
import heapq
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from matplotlib.pyplot import figure
import numpy as np
import cv2
from utils import pixels_conversion, enum_to_unit, to_coord_list
import pandas as pd
from typings import Unit, Workflow, DataObj, OutputOptions, WorkflowObj
from typing import List, Tuple
import time
from PyQt5.QtCore import pyqtSignal
import logging

def points2line(X, Y):
    lines = []
    previousVal = 3 # X = 0 ---- Y = 1
    coords = list(zip(X, Y))
    new_coords = coords.copy()
    lines.append([new_coords[0]])

    for x in range(1, len(new_coords)):
        if(new_coords[x][0] == coords[x - 1][0]): # If current point and previous point share X coordinte
            dirVal = 0  # Set our direction variable to X
        else:
            dirVal = 1  # Set our direction variable to Y

        if(dirVal == previousVal):    # If the current and previous directions are the same
            lines[-1].append(coords[x]) # Add this point to our current path
        elif(previousVal == 3):       # If this is the first value in our list
            lines[-1].append(coords[x]) # Add this point to our current path
            previousVal = dirVal        # Change direction variable to new direction
        else:                         # If the directions are NOT the same
            lines.append([coords[x]])   # Create a new path
            if(new_coords[x] != new_coords[-1]):  # If we are on last point, no need to check for next point
                if(new_coords[x][0] == coords[x + 1][0]): # If the current and NEXT point share an X value
                    previousVal = 0         # Change direction variable to X
                else:
                    previousVal = 1         # Change direction variable to Y

    midPoints = []
    for x in range(0, len(lines) - 1):
        a_x = (lines[x][-1][0] + lines[x + 1][0][0]) / 2
        a_y = (lines[x][-1][1] + lines[x + 1][0][1]) / 2
        midPoints.append((a_x, a_y))
        
    midPoints.insert(0, lines[0][0])
    midPoints.insert(len(midPoints), lines[-1][-1])

    total_dist = 0
    for i in range(len(midPoints)-1):
        dist = math.sqrt((midPoints[i][0] - midPoints[i+1][0])** 2 + (midPoints[i][1] - midPoints[i+1][1])** 2)
        total_dist += dist

    return(midPoints, total_dist)

def map_downscale(img_path: str, mask_path: str):
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

    new_grid = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(grid))))
    new_grid_flipped = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(flip_grid))))

    return(new_grid, new_grid_flipped)

def map_fill(img, flip = True):
    # img = cv2.imread(img_path, 0)

    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    if(flip == True):
        binary = cv2.bitwise_not(binary)

    out = np.zeros_like(binary)

    contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, 255, cv2.FILLED)

    out[out == 0] = 1
    out[out == 255] = 0

    return(out)

def run_astar(map_path, mask_path, coord_list: List[Tuple[float, float]], alt_list: List[Tuple[float, float]]):
# def run_astar(map_path, mask_path, coord_list: List[Tuple[float, float]], alt_list: List[Tuple[float, float]], pb: pyqtSignal):
    """ RUN ASTAR ON A MAP """

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
        result = [[-1 for i in range(no_rows)] for j in range(no_columns)]
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

    def search(maze, cost, start, end, its = None, cutoff = True):
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
            # print(current_node.position)
            current_index = 0
            for index, item in enumerate(yet_to_visit_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
                    
            # if we hit this point return the path such as it may be no solution or 
            # computation cost is too high
            if outer_iterations >= max_iterations:
                # print ("giving up on pathfinding too many iterations")
                # print("Iterations: " + str(outer_iterations))
                if(cutoff):
                    return
                else:
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
                # print(f"{node_position[1]}, {node_position[0]}, {maze[node_position[1]][node_position[0]]}")

                # Make sure within range (check if within maze boundary)
                if (node_position[1] > (no_rows - 1) or 
                    node_position[1] < 0 or 
                    node_position[0] > (no_columns -1) or 
                    node_position[0] < 0):
                    continue

                # Make sure walkable terrain
                if maze[node_position[1]][node_position[0]] != 0:
                    continue

                # Create new node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                
                # Child is on the visited list (search entire visited list)
                if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                    # print("Child on visited list...")
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
            if outer_iterations >= max_iterations:
                # print ("giving up on pathfinding too many iterations")
                # print("Iterations: " + str(outer_iterations))
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
                # print(f"{node_position[1]}, {node_position[0]}, {maze[node_position[1]][node_position[0]]}")

                # Make sure within range (check if within maze boundary)
                if (node_position[1] > (no_rows - 1) or 
                    node_position[1] < 0 or 
                    node_position[0] > (no_columns -1) or 
                    node_position[0] < 0):

                    # print("OOB")
                    continue

                # Make sure walkable terrain
                if maze[node_position[1]][node_position[0]] != 0:
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

    def makePath(maze, cost, start, end, its = None, cutoff = True):
        path = search(maze, cost, start, end, its, cutoff)

        if(path == None):
            # print("No Values Found (makePath)")
            # return(None, None)
            return([], [])

        # print(path)

        uVals = []
        for _path in path:
            uVals.extend(unique(_path))


        uVals = list(filter(lambda val: val != -1, uVals))

        uArray =  np.asarray(path)
        orderedVals = []
        xVals = []
        yVals = []

        for i in range(len(uArray)):
            for j in range(len(uArray[i])):
                if(uArray[i][j] in uVals):
                    orderedVals.append([uArray[i][j], i, j])

        orderedVals = sorted(orderedVals)

        for i in orderedVals:
            xVals.append(i[1])
            yVals.append(i[2])
            # print(f"{i[1]}, {i[2]}")

        return xVals, yVals

    def makeHolePath(maze, cost, start, end, its = None):
        path = holeSearch(maze, cost, start, end, its)
        # print(path)

        if(path == None):
            # print("No Values Found (makeHolePath)")
            return(None, None)

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

        return xVals, yVals

    def getDist(p1, p2):
        x_i, y_i = p1
        x_j, y_j = p2

        dist = math.sqrt(((x_j - x_i) ** 2) + ((y_j - y_i) ** 2))

        return(dist)

    def orderCoords(test_coordinate: Tuple[float, float], alt_coordinate_list: List[Tuple[float, float]]):
        dist_list = []

        # p1 = (test_coordinate[1], test_coordinate[0])
        p_if_y, p_if_x = test_coordinate

        for j in alt_coordinate_list:
            p2 = (j[1], j[0])
            if test_coordinate != p2:
                p_jf_y, p_jf_x = p2
                dist = math.sqrt(((p_jf_y - p_if_y) ** 2) + ((p_jf_x - p_if_x) ** 2))
                dist_obj = [p_jf_y, p_jf_x, dist]
                dist_list.append(dist_obj)
        dist_list.sort(key = lambda x: x[2])

        # for x in dist_list:
            # print(x)

        return(dist_list)

    map, flippedMap = map_downscale(map_path, mask_path)

    map = map_fill(map)
    flippedMap = map_fill(flippedMap, False)
    
    # If you remove this comment the code will blow up
    
    def astar(_coord_list: List[Tuple[float, float]], _alt_list: List[Tuple[float, float]]):
        logging.info("initializing variables & making maps...")

        i_l = 0         # Initialize particle counter
        nnd_list = []   # Create blank list to store particle, matching landmark, and distance
        step_list = []  # Create ANOTHER blank list to store coordinates

        for i, particle in enumerate(_coord_list):  # Downscale landmark by 1/16th coordinates to match map downscaling
            _coord_list[i] = (int(particle[0]), int(particle[1]))

        for i, alt_coord in enumerate(_alt_list):   # Downscale landmark by 1/16th coordinates to match map downscaling
            _alt_list[i] = (int(alt_coord[0] * (1/16)), int(alt_coord[1] * (1/16)))
        
        logging.info("running A* algorithm...")

        start_time = time.time()                            # Start stopwatch at beginning of program
        lastTime = start_time                               # Create initial lastTime

        for p in _coord_list:
            i_l += 1                                        # Increment particle counter
            print(f"Particle: {i_l}")
            i_j = 0                                         # Counter for each landmark (per particle)
            small_dist = 10000000000000000000               # Initial value for smallest dist (will be overidden)
            max_len = 1000                                  # Initial value for pathfinding cutoff
            p1 = (int(p[0]/16), int(p[1]/16))               # Downscale particle by 1/16th coordinates to match map downscaling
            nnd_obj = [(p[0], p[1]), (0,0), 0, 0, 0, 0]     # Create blank object to store value we want to push to the exported DataFrame
            # ^^^ Particle, Landmark, G* Dist, A* Dist, Smooth Dist
            path_coords = [(p[0], p[1]), 0]                            # ^^^ Same as above but with path coordinates
            ordered_alt_list = orderCoords(p1, _alt_list)   # Order landmarks to check by shortest to longest length from particle

            for j in ordered_alt_list:
                i_j += 1                # Increment landmark counter

                p2 = (j[0], j[1])       # Get current landmark coordinates

                if p1 != p2:            # Make sure landmark and particle aren't the same...
                   
                    c_map = map.copy()  # Create copy of map

                    if(map[p2[1]][p2[0]] == 1):
                        _Xi, _Yi = makeHolePath(flippedMap, 1, (p2[0], p2[1]), (p1[0], p1[1]), (max_len - 1))
                        _XYi = np.column_stack((_Xi,_Yi))
                        for x in _XYi:
                            c_map = cv2.circle(c_map, x, 5, (0, 0, 0), -1)

                    if(map[p1[1]][p1[0]] == 1):
                        print("out")
                    # If our particle is outside of the mask area...
                        _Xi, _Yi = makeHolePath(flippedMap, 1, (p1[0], p1[1]), (p2[0], p2[1]), (max_len - 1))
                        # In a similar way to our landmark path, map from our particle to a collision point
                        # This code functions identically to the code above, except for when we calculate our connecting path

                        _XYi = np.column_stack((_Xi,_Yi))
                        for x in _XYi:
                            c_map = cv2.circle(c_map, x, 5, (0, 0, 0), -1)

                        _X, _Y = makePath(c_map, 1, (p1[0], p1[1]), (p2[0], p2[1]), ((max_len - 1)), True)
                        
                        # Aaaaand here's the connecting path! (between our two collision points)

                        dist = len(_Y)
                    else:
                    # If our particle is inside of the mask area...
                        _X, _Y = makePath(c_map, 1, (p1[0], p1[1]), (p2[0], p2[1]), ((max_len - 1)), True)
                        # Make our connecting path from the particle to our landmark collision point

                        dist = len(_Y)

                    # '''
                    fig = plt.figure()
                    plt.scatter(_X, _Y)
                    plt.plot(p2[0], p2[1], 'y*')
                    plt.plot(p1[0], p1[1], 'r*')
                    plt.imshow(c_map, cmap='binary')
                    # plt.show()
                    plt.savefig(f"review_imgs/{i_l}_{i_j}.png")
                    # '''

                    if(_Y == []): # Make sure our paths exist
                        continue

                    mids, smoothDist = points2line(_X, _Y)

                    curTime = round(time.time() - start_time, 2)    #Get current time since program start
                    timeSince = round(curTime - lastTime, 2)        #Get time since last loop
                    lastTime = curTime                              #Set loop timer to current time

                    print(f"TIME: {curTime} Particle: {i_l}, Landmark: {i_j} - {smoothDist}", {map[p2[1]][p2[0]]})

                    xs = [x[0] for x in mids]
                    ys = [x[1] for x in mids]

                    '''
                    fig = plt.figure()
                    plt.scatter(_X, _Y)
                    plt.plot(p2[0], p2[1], 'y*')
                    plt.plot(p1[0], p1[1], 'r*')
                    plt.imshow(c_map, cmap='binary')
                    plt.show()
                    # '''

                    if dist < small_dist:           # If we find a new smallest distance...
                        if(j[2] <= smoothDist):
                            # print("New Smallest Dist!") 
                            small_dist = dist           # Make sure we update the smallest distance to the new one
                            max_len = dist              # As well as updating the maximum length of our paths
                                                        # (^^ this is to cut down on runtime)
                            # Updating our arrays:
                            nnd_obj[1], nnd_obj[2], nnd_obj[3], nnd_obj[4], nnd_obj[5] = (p2[0]*16, p2[1]*16), j[2]*16, small_dist*16, smoothDist*16, smoothDist*16
                            path_coords[1] = [tuple(x * 16 for x in point) for point in mids]
                            path_coords[1][0] = nnd_obj[0]
                        else:
                            # print(f"{j[2]}, {smoothDist}")
                            logging.info("Aborting...")
            
            # logging.info("Appending Data...")
            nnd_list.append(nnd_obj)        # Add the finalized match to our DataFrame
            step_list.append(path_coords)   # Add the finalized coordinates to our DataFrame
        logging.info("Returning Data...")
        return nnd_list, step_list          # DONE!!

    logging.info("Creating Dataframes...")
    real_astar_list, astar_coords = astar(coord_list, alt_list)
    real_df = pd.DataFrame(data={'Nearest Neighbor A* Distance': real_astar_list})

    clean_real_df = pd.DataFrame()
    clean_real_df[['og_coord', 'astar_coord', 'goldstar_dist', 'astar_dist', 'smoothed_dist', 'dist']] = pd.DataFrame(
        [x for x in real_df['Nearest Neighbor A* Distance'].tolist()])
    # clean_real_df = clean_real_df.sort_values(['dist'])

    # df = pd.DataFrame(astar_coords, columns=["a*Y", "a*X"])
    df = pd.DataFrame(astar_coords, columns=['og_coord', 'Path'])
    logging.info("Finishing...")
    return clean_real_df, clean_real_df, df, df

def draw_astar(nnd_df: pd.DataFrame, bin_counts: List[int], img: List, palette: List[Tuple[int, int, int]], circle_c: Tuple[int, int, int] = (0, 0, 255)):
    """ DRAW LINES TO ANNOTATE N NEAREST DIST ON IMAGE """
    def sea_to_rgb(color):
        color = [val * 255 for val in color]
        return color

    count, bin_idx = 0, 0
    for idx, entry in nnd_df.iterrows():
        count += 1
        particle_1 = tuple(int(x) for x in entry['og_coord'])
        particle_2 = tuple(int(x) for x in entry['astar_coord'])
        if count >= bin_counts[bin_idx] and bin_idx < len(bin_counts) - 1:
            bin_idx += 1
            count = 0
        img = cv2.circle(img, particle_1, 10, circle_c, -1)
        img = cv2.line(img, particle_1, particle_2, sea_to_rgb(palette[bin_idx]), 5)
        img = cv2.circle(img, particle_2, 10, (0, 0, 255), -1)

    for idx, entry in nnd_df.iterrows():
        particle_1 = tuple(int(x) for x in entry['og_coord'])
        cv2.putText(img, str(idx), org=particle_1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=0.5)
    return img