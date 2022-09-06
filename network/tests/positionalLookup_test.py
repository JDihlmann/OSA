import os 
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import tensorflow_probability as tfp
tfd = tfp.distributions

module_path = str(Path(__file__).parent.parent.parent.resolve())
if module_path not in sys.path:
    sys.path.append(module_path)


from network.network import Network
from network.datasetLoader.avatarDatasetLoader import AvatarDatasetLoader


network = Network(None, None)

old_grid = network.grid

# tensorflow
def mnd_pdf(x, mu, sigma):
    return tf.exp(-0.5 )

print(network.fNetwork.model.summary())

mu = [1,0,0]
sigma = [0.1,0.1, 1]
grid = tf.cast(old_grid, tf.float32)
mnfc = tfd.MultivariateNormalDiag(mu, sigma) 
grid = mnfc.prob(grid) 


grid = tf.repeat(grid, 3, axis=2)
grid = tf.reshape(grid, [512,512,256,3])

grid = grid.numpy()
grid[:, :, :, 1] = (old_grid[:, :, :, 2] +1.0) / 2.0
grid[:, :, :, 2] = 0

step_resolution = 8

x = np.arange(start=0, stop=512, step=step_resolution).repeat(512*256 // step_resolution)
y = np.tile(np.arange(start=0, stop=512, step=step_resolution).repeat(512  // step_resolution), 256)
z = np.tile(np.arange(start=0, stop=256, step=step_resolution), (512*512 // step_resolution))
v = grid[x,y,z,:] / np.max(grid[x,y,z,:])

print(v[v[:,0] > 0.01,0])

colors = (v *255).astype(int)
points = np.stack([x,y,z], axis=1)

# get colors greater than 0.5
colors[v[:,0] < 0.01, :] = 255


avatar_dataset = AvatarDatasetLoader()
avatar_dataset._show_avatar_points(points, colors)