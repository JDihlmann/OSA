from importlib.resources import path
import os
import bpy
import sys
import math 
import json
import random
import numpy as np
from pathlib import Path
from enum import Enum
from mathutils import Vector 

# load custom modules
sys.path.append(os.getcwd())
from utils.loadAvatar import loadAvatar, AvatarType, avatarTypeConverter
from utils.translateCamera import moveCamera, lookAtPoint
from utils.loadHDR import loadHDR
from utils.deleteObjects import deleteObjects
from utils.translateAvatar import moveAvatar, rotateAvatar, rotateAvatarBones

# delete default cube 
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# save path 
avatarLibraryPath = Path(__file__).parent.parent.resolve() / 'datasets' / 'avatars'

# list of all avatars
avatars = []
avatarTypes = [name for name in os.listdir("../datasets/avatars") if os.path.isdir("../datasets/avatars/" + name)]
for avatarType in avatarTypes:
    avatarTypeDir = "../datasets/avatars/" + avatarType
    avatarNames = [name for name in os.listdir(avatarTypeDir) if os.path.isdir(avatarTypeDir + "/" + name)]
    
    for avatarName in avatarNames:
        avatars.append((avatarName, avatarTypeConverter(avatarType)))

selectedAvatar = avatars
# selectedAvatar = [avatars[0]]


for i in range(0, len(selectedAvatar)):
    avatar = selectedAvatar[i]

    # avatar
    avatarName = avatar[0]
    avatarType = avatar[1]
    avatarObjects = loadAvatar(avatarName, avatarType.value)
    avatarBody = None
    for obj in avatarObjects:
        if  "Bip01.001" == obj.name or "Bip02" == obj.name:
            avatarBody = obj
            break
    

    print("Exporting avatar: " + avatarName)
    file_path = str(avatarLibraryPath / avatarType.value / avatarName) + "/" + avatarName + ".obj" 

    bpy.ops.export_scene.obj(filepath=file_path, use_materials=True)

    # delete objects
    deleteObjects(avatarObjects)

