"""
Program to find year round orienting of seasons
"""

import math
from pQueue import PriorityQueue
from PIL import Image


LATITUTE = 10.29
LONGITUDE = 7.55


SEASONS= ("summer","fall","winter","spring")
TRAIL= ("white", "brown", "red")

START_X, START_Y = 0,0
END_X, END_Y = 394,499

PATH = (255,0,0,255)

Land=(248,148,18,255)
Ice=(123,255,255,255)
R_meadow=(255,192,0,255)
E_forest=(255,255,255,255)
S_forest=(2,208,60,255)
W_forest=(2,136,40,255)
Vegetation=(5,73,24,255)
Water=(0,0,255,255)
Road=(71,51,3,255)
Foot_path=(0,0,0,255)
Mud=(141,76,0,255)
Out_of_bounds=(205,0,101,255)

def load_map():
    im = Image.open("terrain.png")
    return im

def load_elevation_matrix():
    """
    Elevation matrix to read the elevation text
    :return: the values of elevated matrix
    """
    with open("mpp.txt") as f:
        elevation_matrix = [elevation_matrix.split() for elevation_matrix in f]
    for i in range(5):
        for row in elevation_matrix:
            del row[-1]
    return elevation_matrix

def get_checkpoints(filename):
    with open(filename) as f:
        checkpoints = [ tuple(map(int,[x for x in line.split()])) for line in f]
    return checkpoints


TERRAIN_COSTS = {
    Land:1,
    Ice:15,
    R_meadow:45,
    E_forest:20,
    S_forest:55,
    W_forest:75,
    Vegetation:100000,
    Water:1000000,
    Road:1,
    Foot_path:10,
    Out_of_bounds: math.inf,
    Mud: 60
}
ELEVATION = load_elevation_matrix()

def get_euclidian_distance(start, end):
    """
    Runs the heuristic functionality
    :param start: starts reading the coordinates
    :param end: ends the coordinates
    :return: distance as a number
    """
    x1, y1 = start
    x2, y2 = end
    x_distance = abs(x2-x1) * LATITUTE
    y_distance = abs(y2-y1) * LONGITUDE
    return math.sqrt(x_distance ** 2 + y_distance ** 2)

def get_neighbours(current, depth=1):
    """
    Function used to find the next neighbour
    :param current: Current pixel value
    :param depth: The overall search of the pixels
    :return: all the nearest neighbours with closest value
    """
    neighbours = set()
    x, y = current
    if y - depth >= START_Y:
        neighbours.add((x, y-depth))
        if x - depth >= START_X:
            neighbours.add((x-depth, y - depth))
        if x + depth <= END_X:
            neighbours.add((x + depth, y - depth))
    if y + depth <= END_Y:
        neighbours.add((x, y + depth))
        if x - depth >= START_X:
            neighbours.add((x-depth, y + depth))
        if x + depth >= END_X:
            neighbours.add((x+depth, y + depth))
    if x - depth >= START_X:
        neighbours.add((x - depth, y))
    if x + depth <= END_X:
        neighbours.add((x + depth, y))

    return neighbours

def get_cost(map, current, next):
    """
    gets the cost of the nearest neighbour
    :param map:the entire map is considered
    :param current:cost from current to the nearest neighbour
    :param next:selection of the closest neighbour
    :return:Elevation matrix
    """
    x, y=current
    next_x, next_y = next
    #print(next_y,next_x)
    return abs(float(ELEVATION[next_y][next_x]) - float(ELEVATION[y][x])) * TERRAIN_COSTS[map[next_x,next_y]]

def find_path(map,start,end):
    """
    Function written to find path to the next pixel
    :param map: the entire map is considered
    :param start: starts searching the map
    :param end: goes through each and every pixel to get the final path
    :return: previous and distance from next
    """
    prev, costFromPixel, distanceFromStart = dict(), dict(), dict()
    prev[start] = None
    costFromPixel[start] = 0
    distanceFromStart[start] = 0
    pQueue = PriorityQueue()
    pQueue.put(start, 0)
    while not pQueue.empty():
        current = pQueue.get()

        if current == end:
            break
        for next in get_neighbours(current):
            current_cost = costFromPixel[current] + get_cost(map,current,next)
            isCostInfinite = current_cost == math.inf
            isVisited = next in costFromPixel
            if not isCostInfinite and (not isVisited or current_cost < costFromPixel[next]):
                priority = current_cost + get_euclidian_distance(next, end)
                pQueue.put(next,priority)
                prev[next] = current
                costFromPixel[next] = current_cost
                distanceFromStart[next] = distanceFromStart[current] + get_euclidian_distance(current, next)
    return prev, distanceFromStart[end]

def get_winter_map(map_img):
    #functionality to print winter
    map = map_img.load()
    ice_coords = []
    for i in range(START_X,END_X):
        for j in range(START_Y, END_Y):
            coord = (i,j)
            if is_ice(map,coord):
                ice_coords.append(coord)
    fill_map(map, ice_coords, Ice)
    return map_img


def is_ice(map,start, max_depth=7):
    #uses winter_map to show the blue edges
    x, y = start
    if map[x, y]!= Water:
        return False
    queue = [start]
    visited = {start}
    current_depth = 0
    elements_counter = 1
    next_elements_counter = 0

    while queue and current_depth <= max_depth:
        current = queue.pop(0)
        neighbors = get_neighbours(current).difference(visited)

        next_elements_counter += len(neighbors)
        elements_counter -= 1
        if elements_counter == 0:
            current_depth += 1
            elements_counter = next_elements_counter
            next_elements_counter = 0

        for neighbor in neighbors:
            visited.add(neighbor)
            x, y = neighbor
            if map[x, y] != Water and map[x, y] != Out_of_bounds:
                return True
            queue.append(neighbor)

    return False


def get_spring_map(map_img):
    #functionality to get spring map done
    map = map_img.load()
    mud_coords = []
    for i in range(START_X, END_X):
        for j in range(START_Y, END_Y):
            coord = (i,j)
            if map[i, j] == Water:
                mud_coords.extend(get_mud_coords(map,coord))
    fill_map(map, mud_coords, Mud)

    return map_img

def get_mud_coords(map,start, max_depth = 15):
    #printing the mud for 15 depth around the water_pixel
    x, y = start
    queue = [start]
    visited = set()
    current_depth = 0
    elements_counter = 1
    next_elements_counter = 0

    mud_coords = set()
    while queue and current_depth <= max_depth:
        current = queue.pop(0)
        visited.add(current)
        neighbors = set()

        for possible_neighbour in get_neighbours(current):
            m, n = possible_neighbour
            if possible_neighbour not in visited and map[m, n] != Water:
                neighbors.add(possible_neighbour)

        next_elements_counter += len(neighbors)
        elements_counter -= 1
        if elements_counter == 0:
            current_depth += 1
            elements_counter = next_elements_counter
            next_elements_counter = 0

        for neighbor in neighbors:
            visited.add(neighbor)
            i,j = neighbor
            if map[i, j] == Out_of_bounds:
                continue
            if map[i, j] != Water and (float(ELEVATION[j][i]) - float(ELEVATION[y][x])) < 1:
                mud_coords.add(neighbor)
            queue.append(neighbor)
    return mud_coords

def get_coords(points, end):
    #gets the coordinates of till end
    coords = []
    current = end
    while current:
        coords.append(current)
        current = points[current]
    return coords

def modify_costs_for_fall(reset=False):
    value = -10 if reset else 10
    TERRAIN_COSTS[E_forest] += value
    TERRAIN_COSTS[S_forest] += value
    TERRAIN_COSTS[W_forest] += value

def fill_map(map, coords, color):
    #fills map with color for the respective seasons
    for x, y in coords:
        map[x, y] = color


def get_map(season):
    #the map is shown to different seasons
    map_img = load_map()
    if season == "summer" or season == "fall":
        return map_img
    elif season == "winter":
        return get_winter_map(map_img)
    elif season == "spring":
        return get_spring_map(map_img)

def process_map(map_img, season, trail):
    """
    does processing for the map
    :param map_img:
    :param season:
    :param trail:
    :return:
    """
    map = map_img.load()
    checkpoints = get_checkpoints(trail + ".txt")
    total_cost = 0
    path_coords = []
    if season == "fall": modify_costs_for_fall()
    for i in range(len(checkpoints)-1):
        start = checkpoints[i]
        end = checkpoints[i+1]
        path_from, cost = find_path(map, start, end)
        total_cost += cost
        path_coords += get_coords(path_from, end)
    fill_map(map, path_coords, PATH)
    if season == "fall": modify_costs_for_fall(reset=True)
    return map_img, total_cost

def main():
    """
    Main function to run the code
    :return: The printing statements
    """
    for season in SEASONS:
        print("We are processing the map for", season, "season!!!!!!!")
        map_img = get_map(season)
        for trail in TRAIL:
            print("Finding path in", season, "map for", trail, "trail")
            map, cost = process_map(map_img.copy(), season, trail)
            map_name = season + "_" + trail + ".png"
            map.save(map_name)
            print("We can see the output in:", map_name)
            print("Path cost for", season,"with", trail, "trail is:", cost, "meters\n")


if __name__ == '__main__':
    main()




