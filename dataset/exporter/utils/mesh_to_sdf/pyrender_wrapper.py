### Wrapper around the pyrender library that allows to
### 1. disable antialiasing
### 2. render a normal buffer
### This needs to be imported before pyrender or OpenGL is imported anywhere

import os
import sys
if 'pyrender' in sys.modules:
    raise ImportError('The mesh_to_sdf package must be imported before pyrender is imported.')
if 'OpenGL' in sys.modules:
    raise ImportError('The mesh_to_sdf package must be imported before OpenGL is imported.')

# Disable antialiasing:
import OpenGL.GL

suppress_multisampling = False
old_gl_enable = OpenGL.GL.glEnable

def new_gl_enable(value):
    if suppress_multisampling and value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)

OpenGL.GL.glEnable = new_gl_enable

old_glRenderbufferStorageMultisample = OpenGL.GL.glRenderbufferStorageMultisample

def new_glRenderbufferStorageMultisample(target, samples, internalformat, width, height):
    if suppress_multisampling:
        OpenGL.GL.glRenderbufferStorage(target, internalformat, width, height)
    else:
        old_glRenderbufferStorageMultisample(target, samples, internalformat, width, height)

OpenGL.GL.glRenderbufferStorageMultisample = new_glRenderbufferStorageMultisample

import pyrender

# Render a normal buffer instead of a color buffer
class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            shaders_directory = os.path.join(os.path.dirname(__file__), 'shaders')
            self.program = pyrender.shader_program.ShaderProgram(os.path.join(shaders_directory, 'mesh.vert'), os.path.join(shaders_directory, 'mesh.frag'), defines=defines)
        return self.program


def render_normal_and_depth_buffers(mesh, camera, camera_transform, resolution):
    global suppress_multisampling
    suppress_multisampling = True
    lightStrength = 1.0
    scene = pyrender.Scene(ambient_light=[lightStrength, lightStrength, lightStrength])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth = False))
    scene.add(camera, pose=camera_transform)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    #renderer._renderer._program_cache = CustomShaderCache()

    # render color
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)

    #render normals
    renderer._renderer._program_cache = CustomShaderCache()
    normal, depth = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    #viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
    suppress_multisampling = False

    return color, depth, normal