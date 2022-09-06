# OSA

> Open sourcing 3D avatar reconstruction from a single image



https://user-images.githubusercontent.com/9963865/184825844-23bbe7ef-17d4-4efd-9090-80d5fd432394.mp4



> This README focuses on how run the code for more detailed information please read the [report][report]

The [Phorhum][phorhum] paper from Google showed astonishing results in reconstructing 3D avatars from a single image.
This work tries to implement their proposed architecture with TensorFlow in an open source fashion.
It features a dataset creator, the reimplemented network architecture, a trained model, and a point cloud viewer.
While the results are far from the one Google showed, this could be used as a starting point to build upon.
This was part of a two-month research internship at the university of tübingen, the human-computer interaction department.

## Dataset

![imageDataset](https://user-images.githubusercontent.com/9963865/184534332-1d2bc88c-42b6-47ae-abeb-e30714161b88.png)

To solve the task of predicting the surface of a single image, a image dataset containing a human and a signed distance field point cloud dataset with color and normal information is needed.
We built the datasets from scratch since there aren't any ready-to-use datasets available.
We use the Microsoft [Rocketbox][rocketbox] avatar dataset as a starting point.
The image dataset was constructed by rendering the avatar models from the avatar dataset within environments that are lit with HDRs.
For the SDF dataset we collected 1 Mio. points per avatar in the split into 500k near points sampled on the mesh and 500k far points sampled around the mesh and the unit sphere.

### Dataset Content

```
.
├── dataset
│   ├── exporter                     # Tools for dataset creation
│   │   ├── ... .py                  # Python scripts that create the dataset
│   │   └── ... .sh                  # Corrseponding bash scripts that call python scripts
│   │
│   ├── loader                       # Tools for dataset loading
│   │   ├── imageDatasetLoader.py    # Image loaded with plotting
│   │   └── avatarDatasetLoader.py   # Avatar OBJ and SDF loader with plotting
│   │
│   └── datasets
│       ├── hdrs                     # HDR dataset
│       ├── sdfs                     # SDF dataset
│       ├── images                   # Image dataset
│       ├── avatars                  # Avatar dataset
│       └── environments             # Environment dataset
.
```

### Built Dataset

A step by step guide to create all the needed datasets (HDRs, SDFs, images, avatars and environments).
The final datasets `sdfs` and `images` are needed to train the model.
Contact me if you want to have my dataset (it was to big to upload it to GitHub).

#### General

- Download the [Rocketbox][rocketbox] avatar dataset.
- Copy `Adults`, `Children`, and `Professions` into `avatars`

#### SDF Dataset

- Create avatar OBJ dataset by running the modified [Mesh2][mesh2sdf] library by running `createOBJDataset.sh
- Find SDF dataset of `near` and `far` points of each avatar in `sdfs`

#### Image Dataset

- Download HDRs from [Polyheaven][polyhaven] and copy them into the `hdrs` directory
- Find environments of 3D photogrammetry scanned scenes from [Sketchfab][sketchfab] and download them
- Preprocess environments by creating a blender scene for each environment with the center of the floor at (0,0,0)
- Store blender scenes in `environments` with a subdirectory for each scene variation
- (Optional) change avatar and camera augmentation settings in `avatarImageExporter.py`
- Run avatar image dataset creation with `createImageDataset.sh`
- Find avatar image dataset in `images`

## Network

![architecture](https://user-images.githubusercontent.com/9963865/184536177-9a7b1669-9174-4936-bfef-b88dac65f3b8.png)

For solving the task of inferring the surface and its color from a single image, we use an end-to-end learnable neural network model, that is inspired by [Phorhum][phorhum].
Given the time and computational constraints of the project, we couldn’t reproduce the full model and used subsets of their implementation.
In the following you find our implementaion with modifications such as the surface projection loss.
In the [report][report] we propose an attention lookup that you can find in the `previous` directory.

### Network Content

```
.
├── network
│   ├── customLayer                  # Custom 3rd party keras layers
│   ├── previous                     # Previous network implementations (including attention)
│   ├── tests                        # Test for losses and custom layers
│   ├── featureExtractorNetwork.py   # Implementation of the feature extractor network G
│   ├── geomertyNetwork.py           # Implementation of the geometry network f
│   ├── loss.py                      # Cusotom losses (including surface projection)
│   └── network.py                   # End to end network with training and inference
.
```

## Train

![random_points](https://user-images.githubusercontent.com/9963865/184547779-4766a7e3-971a-4d4c-b493-69f1d546f74f.png)

Alltought training results are far from the results Google provides, the network does learn some kind of 3D avatar structure.
Sadly color and detailed geometry can not be reconstructed.
By examine the results more closely one could state that there is an issue within the feature extractor network and the network is not able to infer color and geometry information from the images.

### Train Content

```
.
├── train
│   ├── logs                         # Tensorboard logs
│   │
│   ├── models                       # Previous trained models
│   │   ├── f                        # Models for feature extractor network
│   │   └── g                        # Models for geometry network
│   │
│   ├── train.ipynb                  # Start and configure training jupyter notebook
│   └── train.py                     # Start and configure training python script
.
```

### Train Network

The network can be trained by executing either `train.ipynb` or `train.py`.
We trained the network on a machine with 45 GiB RAM, 8 CPUs, and an A6000 GPU with 48GiB for roughly 2 hours for about 6200 steps.

## Viewer

<img width="1884" alt="viewer" src="https://user-images.githubusercontent.com/9963865/184668739-13fb7e61-1157-4d76-8366-3609c95a1583.png">

For visualization purposes, a custom real-time 3D viewer was built rendering millions of points efficiently and enabling the developer to better identify prediction errors.
A client-server architecture was chosen with the server running a Flask application directly interacting with the React Three Fiber client.

### Viewer Content

```
.
├── viewer
│   ├── react                         # React three fiber client
│   └── app.py                        # Flask server
.
```

### Run Viewer

- Choose the correct model in `app.py`
- Start the flask server with `flask run` in directory `viewer`
- Run react three fiber client by calling `yarn dev` in directory `react`

## Misc

This project was part of a research internship at the [human-computer interaction][hci] department by the university of tübingen.
Big thanks to [Efe Bozkir][efe] for his help and mentorship along the project and [Timo Alldieck][timo] and his colleagues for his amazing work on [Phorhum][phorhum].

<!-- Markdown link & img dfn's -->

[phorhum]: https://phorhum.github.io/
[hci]: https://www.hci.uni-tuebingen.de/chair/
[efe]: https://www.hci.uni-tuebingen.de/chair/team/efe-bozkir
[timo]: https://twitter.com/thiemoall?lang=de
[rocketbox]: https://github.com/microsoft/Microsoft-Rocketbox
[mesh2sdf]: https://github.com/marian42/mesh_to_sdf
[polyhaven]: https://polyhaven.com/
[sketchfab]: https://sketchfab.com/
[report]: OSA.pdf
