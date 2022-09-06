import imp


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .customLayers.fourierFeatures import FourierFeatureProjection

class LookupNetwork:
    # 8x fully connected layers with 512 neurons each
    # Swish activation function
    # Ouput of sigmoid for color
    # Ouput of tanh for depth

    
    def __init__(self, input_shape=(3+256)):
        inputs = layers.Input(input_shape)

        split = layers.Lambda(lambda x: tf.split(x, [3, 256], axis=1))(inputs)
        
        x_1 = FourierFeatureProjection(gaussian_projection = 256, gaussian_scale = 1.0)(split[0])
        x = layers.concatenate([x_1, split[1]], axis=1)

        # eight fully connected layer with swish activation and 
        x = layers.Dense(256, activation = keras.activations.swish)(x)
        x = layers.Dense(128, activation = keras.activations.swish)(x)
        x = layers.Dense(64, activation = keras.activations.swish)(x)
        
        # output layer
        output_mu = layers.Dense(2, activation = "tanh")(x)
        output_sigma = layers.Dense(2, activation = "ReLU")(x)
        output_scalar = layers.Dense(1, activation = "linear")(x)

        self.model = keras.Model(inputs, [output_scalar, output_mu, output_sigma], name="LookupNetwork")


if __name__ == "__main__":
    LookupNetwork().model.summary()

