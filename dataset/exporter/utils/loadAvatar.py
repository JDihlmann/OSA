from enum import Enum
from pathlib import Path


avatarLibraryPath = Path(__file__).parent.parent.parent.resolve() / 'datasets' / 'avatars'


class AvatarType(Enum):
    Adult = "Adults"
    Child = "Children"
    Profession = "Professions"

def avatarTypeConverter(avatarType):
    if avatarType == AvatarType.Adult.value:
        return AvatarType.Adult
    elif avatarType == AvatarType.Child.value:
        return AvatarType.Child
    elif avatarType == AvatarType.Profession.value:
        return AvatarType.Profession

def loadAvatar(name, category):
    import bpy
    avatarFile = name + '.fbx'
    avatarPath = avatarLibraryPath / category / name /'Export' / avatarFile
    texturePath = avatarLibraryPath / category / name /'Textures' 

    old_objs = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(
        filepath= str(avatarPath), 
        automatic_bone_orientation= True
    )
    imported_objs = set(bpy.context.scene.objects) - old_objs



    # fix paths to textures
    for im in bpy.data.images:
        if name in im.filepath:
            im.filepath = str(texturePath / im.filepath.split('/')[-1])

    # workaround: make opacity texture fully transparent
    for mat in bpy.data.materials:
        if 'opacity' in mat.name:
            mat.blend_method = 'CLIP'
            mat.alpha_threshold = 1

    return imported_objs
