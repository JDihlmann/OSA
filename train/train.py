import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from dataset.loader.avatarDatasetLoader import AvatarDatasetLoader
from dataset.loader.imageDatasetLoader import ImageDatasetLoader
from network.network import Network

# Load Datasets
image_dataset = ImageDatasetLoader(batch_size=1) # Batch size is 8 for testing
image_dataset.load_data()

position_dataset = AvatarDatasetLoader()
position_dataset.load_data()

# Create Network 
network = Network(image_dataset, position_dataset)
network.train(epoch_count=1)
