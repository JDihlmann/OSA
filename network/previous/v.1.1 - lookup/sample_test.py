
import numpy as np
from datasets.utils.sampleAvatar import sampleAvatar
from network.datasetLoader.imageDatasetLoader import ImageDatasetLoader
from network.datasetLoader.avatarDatasetLoader import AvatarDatasetLoader

image_dataset = ImageDatasetLoader()
avatar_dataset = AvatarDatasetLoader()


path = "/Users/jdihlmann/Documents/OSA/datasets/avatars/Adults/Female_Adult_01/Female_Adult_01.obj"
points, sdf, gradients, colors = sampleAvatar(path, 20000, near_points=True)




points1 = np.zeros([1000,3]) + np.random.uniform(low=0, high=0.01, size=(1000,3))
points1[:,1] = points1[:,1] + 1.0
colors1 = np.zeros([1000,3]) 
colors1[:, 0] = 255

points = np.concatenate([points, points1])
colors = np.concatenate([colors, colors1])

# print(np.min(distances))

# colors = colors[sdf > 0] 
# points = points[sdf > 0]

# distances = np.expand_dims(distances, axis=1)
# points = points - distances * normals

# surface_points = avatar_dataset.map_points_to_surface(points, sdf, gradients)

avatar_dataset._show_avatar_points(points, (colors).astype(int))
# avatar_dataset._show_avatar_points(surface_points, ((0.5 + gradients / 2) * 255.0).astype(int))


