import os 
import sys
import tensorflow as tf
import numpy as np

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from loss import color_loss


estimated_color_near = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
estimated_color_far = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

true_color_near = np.array([[[1, 1, 1], [0, 0, 0], [0, 0, 0]],[[0, 0, 0], [0, 0, 1], [0, 0, 0]]])
true_color_far = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

estimated_color_near = tf.convert_to_tensor(estimated_color_near, dtype=tf.float32)
estimated_color_far = tf.convert_to_tensor(estimated_color_far, dtype=tf.float32)
true_color_near = tf.convert_to_tensor(true_color_near, dtype=tf.float32)
true_color_far = tf.convert_to_tensor(true_color_far, dtype=tf.float32)

print(estimated_color_near.shape)
print(estimated_color_far.shape)
print(true_color_near.shape)
print(true_color_far.shape)


loss = color_loss(true_color_near, estimated_color_near, true_color_far, estimated_color_far, lambda_a1=1, lambda_a2=1)
print(loss)