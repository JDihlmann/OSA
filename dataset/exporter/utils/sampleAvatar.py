import trimesh
import numpy as np
from pathlib import Path
from .mesh_to_sdf import sample_sdf_near_surface


def _get_obj(file_path):
		mesh = trimesh.load(file_path, force="mesh")
		return mesh 

def _get_mesh(file_path):
    obj = _get_obj(file_path)
    return obj

def _get_samples( mesh, sample_size, near_points=True):
    points, sdf, gradients, colors = sample_sdf_near_surface(mesh, number_of_points=sample_size, surface_point_method="scan", sign_method='depth', return_gradients=True, near_points=near_points)
    return points, sdf, gradients, colors

def sampleAvatar(file_path, sample_size, near_points=True):
    mesh = _get_mesh(file_path)

    points, sdf, gradients, colors = _get_samples(mesh, sample_size, near_points=near_points)
    points, sdf, gradients, colors = shuffleData(points, sdf, gradients, colors)

    return points, sdf, gradients, colors

def shuffleData(points, sdf, gradients, colors):
    idx = np.arange(points.shape[0])
    np.random.shuffle(idx)
    return (points[idx], sdf[idx],gradients[idx], colors[idx] )