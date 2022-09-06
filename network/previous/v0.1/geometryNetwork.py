from tensorflow import keras
from tensorflow.keras import layers

class GeomertyNetwork:
    # 8x fully connected layers with 512 neurons each
    # Swish activation function
    # Ouput of sigmoid for color
    # Ouput of tanh for depth

    
    def __init__(self, input_shape=(256+3)):
        inputs = layers.Input(input_shape)

        # eight fully connected layer with swish activation and 
        x = layers.Dense(512, activation = keras.activations.swish)(inputs)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        x = layers.Dense(512, activation = keras.activations.swish)(x)
        
        # output layer for depth and three channel color
        output_depth = layers.Dense(1, activation = "tanh")(x)
        output_color = layers.Dense(3, activation = "sigmoid")(x)
        output_normal = layers.Dense(3, activation = "tanh")(x)

        outputs = layers.concatenate([output_depth, output_color, output_normal])

        self.model = keras.Model(inputs, outputs, name="GeomertyNetwork")


if __name__ == "__main__":
    GeomertyNetwork().model.summary()

