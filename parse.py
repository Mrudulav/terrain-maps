from PIL import Image

def load_map():
    im = Image.open("terrain.png")
    return im

def load_elevation_matrix():
    with open("mpp.txt") as f:
        elevation_matrix = [elevation_matrix.split() for elevation_matrix in f]
    for i in range(5):
        for row in elevation_matrix:
            del row[-1]
    return elevation_matrix

def load_checkpoints(filename):
    with open(filename) as f:
        checkpoints = [ tuple(map(int,[x for x in line.split()])) for line in f]
    return checkpoints
