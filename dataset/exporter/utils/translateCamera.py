import bpy
from mathutils import Vector 


# move camera
def moveCamera(x, y, z):
    camera = bpy.data.objects['Camera']
    camera.location = (x, y, z)

# camera look at target
def lookAtPoint(x, y, z):
    camera = bpy.data.objects['Camera']
    location = camera.location
    target = Vector((x, y, z))

    # calculate the direction the camera should look in
    direction = target - location

    # get the quaternion from the direction
    rotation = direction.to_track_quat('-Z', 'Y')

    # set the camera rotation
    camera.rotation_euler = rotation.to_euler()

