import numpy as np 

def convert_points_to_image_coordinate(points, image_size):
    idx_points = np.floor(((0.5 + points[:,0:2] * 0.5) * image_size)).astype(int)
    idx_points[:, 1] = (image_size - 1) - idx_points[:, 1]
    return idx_points