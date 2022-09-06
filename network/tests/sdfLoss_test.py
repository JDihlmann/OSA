import os 
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path

module_path = str(Path(__file__).parent.parent.parent.resolve())
if module_path not in sys.path:
    sys.path.append(module_path)



from network.network import Network


network = Network(None, None)

true_distance_far = np.array([[[1], [0], [0]], [[0], [0], [-1]]])
estimated_distance_far = np.array([[[-1], [0], [0]], [[0], [0], [0]]])

true_distance_far = tf.convert_to_tensor(true_distance_far, dtype=tf.float32)
estimated_distance_far = tf.convert_to_tensor(estimated_distance_far, dtype=tf.float32)


print(true_distance_far.shape)
print(estimated_distance_far.shape)


loss = network.sdf_loss(true_distance_far, estimated_distance_far)
print(loss)