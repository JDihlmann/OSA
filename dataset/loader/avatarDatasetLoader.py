import os
import math
from pathlib import Path

import trimesh
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from .utils.render import renderMesh, renderPoints
from .utils.image import convert_points_to_image_coordinate
import pyrender
from os import listdir
from os.path import isfile, join


class AvatarDatasetLoader:
    def __init__(self, batch_size=256, image_size=512):
        self.image_size = image_size
        self.batch_size = batch_size

        self.data_path_sdf = str(Path(__file__).parent.parent.absolute()) +   '/datasets/sdfs/'
        self.data_path_avatar = str(Path(__file__).parent.parent.absolute()) +   '/datasets/avatars/'

        self.avatar_ds = {"near":{}, "far":{}}

    def load_data(self):

        for distance_type in ['near', 'far']:
            # Load all file names 
            file_names_npz = [f for f in listdir(self.data_path_sdf + distance_type + "/") if isfile(join(self.data_path_sdf + distance_type + "/", f))]

            for file_name in file_names_npz:
                avatar_name = file_name.split('.')[0]
                points, sdf, color, normals = self.get_points(avatar_name, distance_type)
                points, sdf, color, normals = self.shuffle_data(points, sdf, color, normals)
                idx_points = convert_points_to_image_coordinate(points, self.image_size)
                color = color / 255.0

                # create tensorflow dataset
                self.avatar_ds[distance_type][avatar_name] = tf.data.Dataset.from_tensor_slices((
                    tf.convert_to_tensor(idx_points, dtype=tf.int32), 
                    tf.convert_to_tensor(points, dtype=tf.float32), 
                    tf.convert_to_tensor(sdf, dtype=tf.float32), 
                    tf.convert_to_tensor(color, dtype=tf.float32), 
                    tf.convert_to_tensor(normals, dtype=tf.float32),
                ))
                
                self.avatar_ds[distance_type][avatar_name] = self._configure_for_performance(self.avatar_ds[distance_type][avatar_name])

    
    def shuffle_data(self, points, sdf, colors, normals):
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        points = points[idx]
        sdf = sdf[idx]
        colors = colors[idx]
        normals = normals[idx]

        return points, sdf, colors, normals
    
    def _configure_for_performance(self, ds):
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batch_size)            
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def _get_obj(self, file_path):
        mesh = trimesh.load(file_path, force="mesh")
        return mesh 
    
    def _get_mesh(self, avatar_type, avatar_name):
        file_path = self.data_path_avatar + avatar_type + '/' + avatar_name + '/' + avatar_name+ '.obj'
        obj = self._get_obj(file_path)
        return obj

    def get_points(self, avatar_name, distance_type):
        file_path = self.data_path_sdf + distance_type + '/' + avatar_name+ '.npz'
        file = np.load(file_path)
        points = file['points']
        sdf = file['sdf']
        colors = file['colors']
        normals = file['normals']

        return points, sdf, colors, normals

    def plot_avatar(self, avatar_type, avatar_name):
        mesh = self._get_mesh(avatar_type, avatar_name)

        imgFront, _ = renderMesh(mesh, 0, 0)
        imgRight, _ = renderMesh(mesh, math.pi / 2, 0)
        imgBack, _ = renderMesh(mesh, math.pi , 0)
        imgLeft, _ = renderMesh(mesh, math.pi * 3/ 2, 0)

        imgs = [(imgFront, "front"), (imgRight, "right"), (imgLeft, "left"), (imgBack, "back")]
        self._plot_images(imgs)

    def show_avatar(self, avatar_type, avatar_name):
        mesh = self._get_mesh(avatar_type, avatar_name)
        mesh.show()

    def plot_avatar_points_color(self, points, colors):

        imgFront, _ = renderPoints(points, colors, 0, 0)
        imgRight, _ = renderPoints(points, colors, math.pi / 2, 0)
        imgBack, _ = renderPoints(points, colors, math.pi , 0)
        imgLeft, _ = renderPoints(points, colors, math.pi * 3/ 2, 0)

        imgs = [(imgFront, "front"), (imgRight, "right"), (imgLeft, "left"), (imgBack, "back")]
        self._plot_images(imgs)

    def show_avatar_points_color(self, avatar_name, distance_type):
        points, _, colors, _ = self.get_points(avatar_name, distance_type)
        self._show_avatar_points(points, colors)

    def get_batch(self, avatar_name, distance_type, take_size=20):
        idx_points, points, sdfs, colors, normals = (None, None, None, None, None)
        batches = self.avatar_ds[distance_type][avatar_name].take(take_size)
        for batch in batches: 

            if points is None:
                idx_points, points, sdfs, colors, normals = batch
            else :
                idx_points = np.concatenate((idx_points, batch[0]), axis=0)
                points = np.concatenate((points, batch[1]), axis=0)
                sdfs = np.concatenate((sdfs, batch[2]), axis=0)
                colors = np.concatenate((colors, batch[3]), axis=0)
                normals = np.concatenate((normals, batch[4]), axis=0)

        return idx_points, points, sdfs, colors, normals

    def plot_avatar_points_sdf(self, points, sdf, outside=True):
            
        if outside:
            points = points[sdf > 0]
            sdf = sdf[sdf > 0]
        else:
            points = points[sdf < 0]
            sdf = sdf[sdf < 0]

        colors = np.zeros(points.shape)
        max_sdf = np.abs(np.max(sdf))
        min_sdf = np.abs(np.min(sdf))
        
        if outside:
            colors[:, 0] = 1# (sdf - min_sdf) / (max_sdf - min_sdf)
        else:
            colors[:, 2] = 1# (sdf - min_sdf) / (max_sdf - min_sdf)


        imgFront, _ = renderPoints(points, colors, 0, 0)
        imgRight, _ = renderPoints(points, colors, math.pi / 2, 0)
        imgBack, _ = renderPoints(points, colors, math.pi , 0)
        imgLeft, _ = renderPoints(points, colors, math.pi * 3/ 2, 0)

        imgs = [(imgFront, "front"), (imgRight, "right"), (imgLeft, "left"), (imgBack, "back")]
        self._plot_images(imgs)

    def map_points_to_surface(self, points, distance, normal):
        distance = np.expand_dims(distance, axis=1)
        return points - distance * normal

    def plot_avatar_points_normals(self, points, sdf, normals, projectSurface=True):
            
        # convert normal to color
        normal_color = ((0.5 + normals / 2) * 255.0).astype(int)

        # project points to surface
        if projectSurface:
            points = self.map_points_to_surface(points, sdf, normals)


        imgFront, _ = renderPoints(points, normal_color, 0, 0)
        imgRight, _ = renderPoints(points, normal_color, math.pi / 2, 0)
        imgBack, _ = renderPoints(points, normal_color, math.pi , 0)
        imgLeft, _ = renderPoints(points, normal_color, math.pi * 3/ 2, 0)

        imgs = [(imgFront, "front"), (imgRight, "right"), (imgLeft, "left"), (imgBack, "back")]
        self._plot_images(imgs)

    def show_avatar_points_sdf(self, avatar_name, distance_type):
        points, sdf, _, _ = self.get_points(avatar_name, distance_type)

        colors = np.zeros(points.shape)
        colors[sdf < 0, 2] = 0
        colors[sdf > 0, 0] = 1
        self._show_avatar_points(points, colors)

    def _plot_images(self, images): 
        plt.figure(figsize=(12, 4))
        for i in range(len(images)):
            img, label = images[i]
            plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            plt.imshow(img)
            plt.title(label)
            plt.axis('off')
        plt.show()

    def _show_avatar_points(self, points, color):
        cloud = pyrender.Mesh.from_points(points, colors=color)
        scene = pyrender.Scene()
        scene.add(cloud)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)



   