import numpy as np
import tensorflow as tf

def total_loss(geometry_loss, sdf_loss, geoemetric_regularization, color_loss, lambda_l= 0.2, lambda_e= 0.1 ):
    return geometry_loss + lambda_l * sdf_loss + lambda_e * geoemetric_regularization +  color_loss


# increase lambda_g1 from 1.0 to 15.0 over 100k iterations
def geometry_loss(estimated_distance_near, true_normal_near, estimated_normal_near, lambda_g1=1, lambda_g2=0.1):
    estimated_distance_r = tf.reshape(estimated_distance_near, (estimated_distance_near.shape[0], estimated_distance_near.shape[1]))
    normal_distance  = tf.math.reduce_euclidean_norm(true_normal_near - estimated_normal_near, axis=2)
    linear_difference = lambda_g1 * abs_with_grad(estimated_distance_r) + lambda_g2 * normal_distance
    linear_mean = tf.reduce_mean(linear_difference, axis=1)
    geometry_loss = tf.reduce_mean(linear_mean, axis=0)
    return geometry_loss

def sdf_loss(true_distance_far, estimated_distance_far, k):
    true_distance_far = tf.reshape(true_distance_far, estimated_distance_far.shape)
    true_inside_outside_labels = (tf.math.sign(true_distance_far)+ 1.0) * 0.5 # different version center scale
    estimated_inside_outside_labels = tf.math.sigmoid(k * estimated_distance_far)
    binary_crossentropy = - (true_inside_outside_labels * tf.math.log(estimated_inside_outside_labels + 1e-8) + (1 - true_inside_outside_labels) * tf.math.log(1 - estimated_inside_outside_labels + 1e-8))
    bce_loss = tf.reduce_mean( binary_crossentropy, axis=1)
    sdf_loss = tf.reduce_mean(bce_loss, axis=0)
    sdf_loss = sdf_loss[0]
    return sdf_loss


def color_loss(true_color_near, estimated_color_near, true_color_far, estimated_color_far,  lambda_a1=0.5, lambda_a2=0.3):
    color_loss_near =  tf.reduce_mean(tf.math.reduce_euclidean_norm(true_color_near - estimated_color_near, axis=2), axis=1)
    color_loss_far = tf.reduce_mean(tf.math.reduce_euclidean_norm(true_color_far - estimated_color_far, axis=2), axis=1)
    color_loss = tf.reduce_mean(lambda_a1 * color_loss_near + lambda_a2 * color_loss_far, axis=0)
    return color_loss

def geometric_regularization( estimated_nomal_far):
    normal_euclidian = tf.math.reduce_euclidean_norm(estimated_nomal_far, axis=2)
    normal_distance = tf.square(normal_euclidian - 1)
    normal_distance_mean = tf.reduce_mean(normal_distance, axis=1)
    geoemtric_regularization = tf.reduce_mean(normal_distance_mean, axis=0)
    return geoemtric_regularization

@tf.custom_gradient
def abs_with_grad(x):
    y = tf.abs(x);

    def grad(div): # Derivation intermediate value
        g = 1; # Use 1 to make the chain rule just skip abs
        return div*g;

    return y,grad;