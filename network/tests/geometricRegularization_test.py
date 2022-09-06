import os 
import sys
import tensorflow as tf
import numpy as np

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from network import Network


network = Network(None, None)

estimated_nomal_far = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
estimated_nomal_far = tf.convert_to_tensor(estimated_nomal_far, dtype=tf.float32)

print(estimated_nomal_far.shape)


loss = network.geoemetric_regularization(estimated_nomal_far)
print(loss)