import imp


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .customLayers.fourierFeatures import FourierFeatureProjection

class GeomertyNetwork:
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
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        
        # output layer for depth and three channel color
        output_depth = layers.Dense(1, activation = "linear")(x)
        output_color = layers.Dense(3, activation = "linear")(x)
        output_normal = layers.Dense(3, activation = "linear")(x)

        outputs = layers.concatenate([output_depth, output_color, output_normal])

        self.model = keras.Model(inputs, outputs, name="GeomertyNetwork")


if __name__ == "__main__":
    GeomertyNetwork().model.summary()

