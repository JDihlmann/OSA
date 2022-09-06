import os 
import sys
import tensorflow as tf
import numpy as np

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from network import Network


network = Network(None, None)

estimated_distance = np.array([[[-1], [1], [0]], [[0], [0], [0]]])
estimated_normal_far = np.array([[[1, 0, 0], [0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
true_normal_far = np.array([[[0, 0, 0], [1, 0, 0], [-1, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

estimated_distance = tf.convert_to_tensor(estimated_distance, dtype=tf.float32)
estimated_normal_far = tf.convert_to_tensor(estimated_normal_far, dtype=tf.float32)
true_normal_far = tf.convert_to_tensor(true_normal_far, dtype=tf.float32)

print(estimated_distance.shape)
print(estimated_normal_far.shape)
print(true_normal_far.shape)


print(true_normal_far.shape)
loss = network.geometry_loss(estimated_distance, true_normal_far, estimated_normal_far, lambda_g1=1, lambda_g2=1)
print(loss)