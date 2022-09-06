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

# save path 
imageLibraryPath = Path(__file__).parent.parent.resolve() / 'datasets' / 'images'


# file name 
blendFilePath = bpy.data.filepath
blendFileName = blendFilePath.split('/')[-1].split('.')[0]

# set render settings image resolution
bpy.data.scenes['Scene'].render.resolution_x = 512
bpy.data.scenes['Scene'].render.resolution_y = 512

# list of all avatars
avatars = []
avatarTypes = [name for name in os.listdir("../datasets/avatars") if os.path.isdir("../datasets/avatars/" + name)]
for avatarType in avatarTypes:
    avatarTypeDir = "../datasets/avatars/" + avatarType
    avatarNames = [name for name in os.listdir(avatarTypeDir) if os.path.isdir(avatarTypeDir + "/" + name)]
    
    for avatarName in avatarNames:
        avatars.append((avatarName, avatarTypeConverter(avatarType)))

# list of all hdrs
hdrs = [name for name in os.listdir("../datasets/hdrs") if os.path.isfile("../datasets/hdrs/" + name)]

# select avatars
# sa = avatars[20]
# selectedAvatar = [sa,sa,sa,sa,sa]
selectedAvatar = avatars

# camera max position offsets
cameraParams = {
    "position": {
        "x": {"max": 0.1, "min": -0.1},    
        "y": {"max": -2.5, "min": -2.51},      
        "z": {"max": 1.6, "min": 1.59}      
    },
    "target": {
        "x": {"max": 0.05, "min": -0.05},    
        "z": {"max": 1.0, "min": 0.8}     
    },
}

# avatar max position offsets
avatarParams = {
    "position": {
        "x": {"max": 0.05, "min": -0.05},    
        "y": {"max": 0.05, "min": -0.05},      
    },
    "rotation": {
        "z": {"max": -85, "min": -95},      
    },
    "bone": {
        "x": {"max": 5, "min": -5},      
        "y": {"max": 5, "min": -5},      
        "z": {"max": 5, "min": -5},      
    },
}

# save render parameter
renderSettings = {}

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

    avatarX = random.uniform(avatarParams["position"]["x"]["min"], avatarParams["position"]["x"]["max"])
    avatarY = random.uniform(avatarParams["position"]["y"]["min"], avatarParams["position"]["y"]["max"])
    moveAvatar(avatarBody, avatarX, avatarY, avatarBody.location.z)

    avatarRotZ = random.uniform(avatarParams["rotation"]["z"]["min"], avatarParams["rotation"]["z"]["max"])
    rotateAvatar(avatarBody, avatarRotZ)

    avatarBoneX = random.uniform(avatarParams["bone"]["x"]["min"], avatarParams["bone"]["x"]["max"])
    avatarBoneY = random.uniform(avatarParams["bone"]["y"]["min"], avatarParams["bone"]["y"]["max"])
    avatarBoneZ = random.uniform(avatarParams["bone"]["z"]["min"], avatarParams["bone"]["z"]["max"])
    rotateAvatarBones(avatarBody, avatarBoneX, avatarBoneY, avatarBoneZ)

    # camera
    cameraX = random.uniform(cameraParams["position"]["x"]["min"], cameraParams["position"]["x"]["max"])
    cameraY = random.uniform(cameraParams["position"]["y"]["min"], cameraParams["position"]["y"]["max"])
    cameraZ = random.uniform(cameraParams["position"]["z"]["min"], cameraParams["position"]["z"]["max"])
    moveCamera(cameraX, cameraY, cameraZ)
    # moveCamera(0, -5, 1.2)

    # look at target
    targetX = random.uniform(cameraParams["target"]["x"]["min"], cameraParams["target"]["x"]["max"])
    targetZ = random.uniform(cameraParams["target"]["z"]["min"], cameraParams["target"]["z"]["max"])
    lookAtPoint(targetX, 0, targetZ)
    # lookAtPoint(0, 0, 1.6)

    # HDR 
    hdr = random.choice(hdrs)
    loadHDR(hdr)


    
    # change samples
    avatarFileName = avatarName # +  "_" + str(i)
    bpy.context.scene.render.filepath = str(imageLibraryPath / blendFileName / avatarFileName)
    bpy.ops.render.render(write_still=True)

    # save render parameter
    renderSettings[avatarName] = {
        "name": avatarName,
        "camera": {
            "position": {
                "x": cameraX,    
                "y": cameraY,      
                "z": cameraZ      
            },
            "target": {
                "x": targetX,    
                "z": targetZ     
            },
        },
        "hdr": hdr,
        "avatar": {
            "position": {
                "x": avatarX,
                "y": avatarY,
            },
            "rotation": {
                "z": avatarRotZ,
            },
            "bone": {
                "x": avatarBoneX,
                "y": avatarBoneY,
                "z": avatarBoneZ,
            },
        },
    }

    # delete objects
    deleteObjects(avatarObjects)

# save render parameter as json file
with open(str(imageLibraryPath / blendFileName / "info.json"), 'w') as outfile:
    json.dump(renderSettings, outfile)
