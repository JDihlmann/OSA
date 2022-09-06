from importlib.resources import path
import sys
import os
import numpy as np
import numpy as np
from pathlib import Path


from utils.loadAvatar import  AvatarType, avatarTypeConverter
from utils.sampleAvatar import sampleAvatar

avatarType = str(sys.argv[1])
avatarName = str(sys.argv[2])


print("Sampling: " + avatarName )

# save path 
sdfLibraryPath = Path(__file__).parent.parent.resolve() / 'datasets' / 'sdfs'
avatarLibraryPath = Path(__file__).parent.parent.resolve() / 'datasets' / 'avatars'

avatarPath = str(avatarLibraryPath / avatarType / avatarName / avatarName) + '.obj'

# Near points 
points, sdf, gradients, colors = sampleAvatar(avatarPath, sample_size=500000, near_points=True)

# save sdf as npz file
file_path = str(sdfLibraryPath) + "/near/" + avatarName + ".npz" 

np.savez(file_path, points=points, sdf=sdf, colors=colors, normals=gradients)
print("Avatar: " + avatarName + " near saved")

# Far points
points, sdf, gradients, colors = sampleAvatar(avatarPath, sample_size=500000, near_points=False)

# save sdf as npz file
file_path = str(sdfLibraryPath) + "/far/" + avatarName + ".npz" 

np.savez(file_path, points=points, sdf=sdf, colors=colors, normals=gradients)
print("Avatar: " + avatarName + " far saved")
