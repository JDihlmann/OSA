import bpy

def deleteObjects(objects):
    for obj in objects:
        bpy.data.objects.remove(obj)