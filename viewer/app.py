import os
from random import seed
import sys
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from pathlib import Path
from os import listdir
from os.path import isfile, join, isdir

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from dataset.exporter.utils.mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere
from network.network import Network
from dataset.loader.imageDatasetLoader import ImageDatasetLoader
from dataset.loader.avatarDatasetLoader import AvatarDatasetLoader

image_dataset = ImageDatasetLoader()
avatar_dataset = AvatarDatasetLoader()

# create network 
network = Network()
network.load_weights(9)  

app = Flask(__name__)
CORS(app)

def loadTrueAvatar(name, distanceType):
    if(distanceType == 'near'):
        tpoints, tdistances, tcolors, tnormals  = avatar_dataset.get_points(name, "near")
    else:
        tpoints, tdistances, tcolors, tnormals  = avatar_dataset.get_points(name, "far")

    tcolors = tcolors / 255.0
    return {"points": tpoints.tolist(), "colors": tcolors.tolist(), "normals": tnormals.tolist(), "distances": tdistances.tolist()}

def loadEstimatedAvatar(name, environment, distanceType, ):
    image = image_dataset.load_image(environment, name)[:,:,0:3]  

    if(distanceType == 'near'):
        tpoints, tdistances, tcolors, tnormals  = avatar_dataset.get_points(name, "near")
        tpoints = tpoints[0:200000]
        distances, colors, normals = network.inference(image, tpoints)
    else:
        tpoints, tdistances, tcolors, tnormals  = avatar_dataset.get_points(name, "far")
        tpoints = tpoints[0:200000]
        # new code
        points = sample_uniform_points_in_unit_sphere(200000)
        tpoints = points
        distances, colors, normals = network.inference(image, tpoints)

        ''' for i in range(4):
            points = avatar_dataset.map_points_to_surface(points, distances, normals)
            distances, colors, normals = network.inference(image, points) '''
    

    return {"points": tpoints.tolist(), "colors": colors.tolist(), "normals": normals.tolist(), "distances": distances.tolist()}


@app.route("/load/true/near/<string:name>")
@cross_origin()
def loadTrueAvatarNear(name):
    return loadTrueAvatar(name, "near")

@app.route("/load/true/far/<string:name>")
@cross_origin()
def loadTrueAvatrFar(name):
    return loadTrueAvatar(name, "far")

@app.route("/load/estimated/near/<string:name>/<string:environment>")
@cross_origin()
def loadEstimatedAvatarNear(name, environment):
    return loadEstimatedAvatar(name, environment, "near")

@app.route("/load/estimated/far/<string:name>/<string:environment>")
@cross_origin()
def loadEstimatedAvatarFar(name, environment):
    return loadEstimatedAvatar(name, environment, "far")

@app.route("/load/names")
@cross_origin()
def getNames():
    data_path_sdf = str(Path(__file__).parent.parent.absolute()) +   '/dataset/datasets/images/'
    environment_names = [f for f in listdir(data_path_sdf) if isdir(join(data_path_sdf , f))]
    avatar_names = [f.split(".")[0] for f in listdir(data_path_sdf + environment_names[0] + "/") if isfile(join(data_path_sdf + environment_names[0] + "/" , f))]
    response = jsonify({"environments": environment_names, "avatars": avatar_names})
    return response