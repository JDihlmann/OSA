from math import radians
import bpy
from mathutils import Vector 
import numpy as np

# move avatar
def moveAvatar(avatarObject, x, y, z):

    # Clear animation data 
    avatarObject.animation_data_clear()

    # Set the location of the avatar
    avatarObject.location.x = x
    avatarObject.location.y = y
    avatarObject.location.z = z


# rotate avatar
def rotateAvatar(avatarObject, degree):

    # Clear animation data 
    avatarObject.animation_data_clear()

    # rotate
    degree2Rad = degree * (np.pi / 180)
    avatarObject.rotation_euler.z = degree2Rad

# rotate avatar bones
def rotateAvatarBones(avatarObject, x, y, z):
    bonePrefix = 'Bip02' if avatarObject.name  == 'Bip02' else 'Bip01' 

    # Clear animation data 
    avatarObject.animation_data_clear()
    
    euler = Vector((radians(x), radians(y), radians(z)))
    avatarObject.pose.bones[bonePrefix +" R UpperArm"].rotation_mode = 'XYZ'
    avatarObject.pose.bones[bonePrefix +" R UpperArm"].rotation_euler = euler


    euler = Vector((radians(180-x), -radians(180-y), radians(180-z)))
    avatarObject.pose.bones[bonePrefix +" L UpperArm"].rotation_mode = 'XYZ'
    avatarObject.pose.bones[bonePrefix +" L UpperArm"].rotation_euler = euler


