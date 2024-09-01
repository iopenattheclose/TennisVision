import numpy as np

def get_bbox_center(bbox):
    x1,y1,x2,y2 = bbox
    xcenter = int((x1 + x2) / 2)
    ycenter = int((y1 + y2) / 2)
    return xcenter,ycenter


def measure_distance(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    diff = p2 - p1
    squared_dist = np.dot(diff, diff)
    distance = np.sqrt(squared_dist)
    return distance 

