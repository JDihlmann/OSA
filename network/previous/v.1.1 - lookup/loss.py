import numpy as np
import tensorflow as tf

def total_loss(geometry_loss, sdf_loss, geoemetric_regularization, color_loss, surface_projection_loss=None, lambda_l= 0.5, lambda_e= 0.1 ): # original lambda_l= 0.2, lambda_e= 0.1
    total_loss = geometry_loss + lambda_l * sdf_loss + lambda_e * geoemetric_regularization + color_loss
    if surface_projection_loss:
        total_loss = total_loss + surface_projection_loss
    return total_loss

# increase lambda_g1 from 1.0 to 15.0 over 100k iterations
def geometry_loss(estimated_distance_near, true_normal_near, estimated_normal_near, step_count, lambda_g2=0.1):
    estimated_distance_r = tf.reshape(estimated_distance_near, (estimated_distance_near.shape[0], estimated_distance_near.shape[1]))
    normal_distance  = tf.math.reduce_euclidean_norm(true_normal_near - estimated_normal_near, axis=2)
    linear_difference = geometry_loss_schedule(step_count) * abs_with_grad(estimated_distance_r) + lambda_g2 * normal_distance
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
    color_loss_near =  tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(true_color_near, estimated_color_near), axis=2), axis=1)
    color_loss_far = tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(true_color_far, estimated_color_far), axis=2), axis=1)
    color_loss = tf.reduce_mean(lambda_a1 * color_loss_near + lambda_a2 * color_loss_far, axis=0)
    return color_loss

def geometric_regularization( estimated_nomal_far):
    normal_euclidian = tf.math.reduce_euclidean_norm(estimated_nomal_far, axis=2)
    normal_distance = tf.square(normal_euclidian - 1)
    normal_distance_mean = tf.reduce_mean(normal_distance, axis=1)
    geoemtric_regularization = tf.reduce_mean(normal_distance_mean, axis=0)
    return geoemtric_regularization

def geometry_loss_schedule(step_count, max_step_count=20000, lambda_g1_min=1.0, lambda_g1_max=15.0): # original max_step_count=50k
    lambda_g1 = lambda_g1_min + (lambda_g1_max - lambda_g1_min) * step_count / max_step_count
    return lambda_g1

def surface_projection_loss(true_points_far, true_distance_far, estimated_distance_far, true_normals_far, estimated_normal_far):
    true_projected_surface_points = true_points_far - tf.math.multiply(tf.reshape(true_distance_far, [true_distance_far.shape[0], true_distance_far.shape[1], 1]), true_normals_far)
    estimated_projected_surface_points = true_points_far - tf.math.multiply(tf.reshape(estimated_distance_far, [estimated_distance_far.shape[0], estimated_distance_far.shape[1], 1]), estimated_normal_far)
    distance = tf.reduce_mean(tf.math.reduce_euclidean_norm(true_projected_surface_points - estimated_projected_surface_points, axis=2), axis=1)
    surface_projection_loss = tf.reduce_mean(distance, axis=0)
    return surface_projection_loss

@tf.custom_gradient
def abs_with_grad(x):
    y = tf.abs(x);

    def grad(div): # Derivation intermediate value
        g = 1; # Use 1 to make the chain rule just skip abs
        return div*g;

    return y,grad;