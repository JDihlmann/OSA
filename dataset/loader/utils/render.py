import math 
import trimesh
import pyrender
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

def get_rotation_matrix(angle, axis='y'):
    matrix = np.identity(4)
    if hasattr(Rotation, "as_matrix"): # scipy>=1.4.0
        matrix[:3, :3] = Rotation.from_euler(axis, angle).as_matrix()
    else: # scipy<1.4.0
        matrix[:3, :3] = Rotation.from_euler(axis, angle).as_dcm()
    return matrix

def get_camera_transform_looking_at_origin(rotation_y, rotation_x, camera_height=0.8, camera_distance=2.5):
    camera_transform = np.identity(4)
    camera_transform[1, 3] = camera_height
    camera_transform[2, 3] = camera_distance
    camera_transform = np.matmul(get_rotation_matrix(rotation_x, axis='x'), camera_transform)
    camera_transform = np.matmul(get_rotation_matrix(rotation_y, axis='y'), camera_transform)
    return camera_transform

def renderMesh(mesh, rotation_y, rotation_x):
    if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()

    camera = pyrender.PerspectiveCamera(yfov=1, aspectRatio=1.0, znear = 0.1, zfar = 10)
    camera_transform = get_camera_transform_looking_at_origin(rotation_y, rotation_x)
    # camera_transform = get_camera_transform_looking_at_origin(math.pi / 2 ,0)

    lightStrength = 1.0
    scene = pyrender.Scene(ambient_light=[lightStrength, lightStrength, lightStrength])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth = False))
    scene.add(camera, pose=camera_transform)

    renderer = pyrender.OffscreenRenderer(1024, 1024)
    # color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.RGBA  | pyrender.constants.RenderFlags.FLAT ) 
    color, depth = renderer.render(scene) 

    return color, depth

def renderPoints(points, color, rotation_y, rotation_x):
    camera = pyrender.PerspectiveCamera(yfov=1, aspectRatio=1.0, znear = 0.1, zfar = 10)
    camera_transform = get_camera_transform_looking_at_origin(rotation_y, rotation_x, camera_height= 0)
    # camera_transform = get_camera_transform_looking_at_origin(math.pi / 2 ,0)

    lightStrength = 1.0
    scene = pyrender.Scene(ambient_light=[lightStrength, lightStrength, lightStrength])
    scene.add(pyrender.Mesh.from_points(points, colors=color, ))
    scene.add(camera, pose=camera_transform)

    renderer = pyrender.OffscreenRenderer(512, 512)
    # color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.RGBA  | pyrender.constants.RenderFlags.FLAT ) 
    color, depth = renderer.render(scene) 
    return color, depth