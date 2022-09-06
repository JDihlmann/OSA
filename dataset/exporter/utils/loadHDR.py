import bpy
from pathlib import Path

# paths 
environmentLibraryPath = Path(__file__).parent.parent.parent.resolve() / 'datasets' / 'hdrs'

def loadHDR(hdrFile):
    hdrPath = environmentLibraryPath / hdrFile

    # Get the environment node tree of the current scene
    node_tree = bpy.context.scene.world.node_tree
    tree_nodes = node_tree.nodes
    
    for node in tree_nodes:
        if( node.type == 'TEX_ENVIRONMENT'):
            image = bpy.data.images.load(str(hdrPath))
            node.image = image
            break

