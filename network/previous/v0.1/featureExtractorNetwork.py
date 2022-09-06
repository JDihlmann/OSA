from tensorflow import keras
from tensorflow.keras import layers
from .customLayers.blurpool import BlurPool2D

class FeatureExtractorNetwork:
    # Unet archtiecture 
    # 13 Layers with 6 encoding and 6 decoding layers
    # Each layer has a convolutional, batch normalization, and relu activation
    # Using Blurpooling in encoding
    # using Upsampling in decoding
    
    def __init__(self, input_shape=(512, 512, 3)):
        inputs = layers.Input(input_shape)
        
        # Encoder
        x1, p1 = self.encoder_block(inputs, 64)     # (256, 256, 64)
        x2, p2 = self.encoder_block(p1, 128)        # (128, 128, 128)
        x3, p3 = self.encoder_block(p2, 256)        # (64, 64, 256)
        x4, p4 = self.encoder_block(p3, 512)        # (32, 32, 512)
        x5, p5 = self.encoder_block(p4, 512)        # (16, 16, 512)
        x6, p6 = self.encoder_block(p5, 512)        # (8, 8, 512)

        # Bridge
        b = self.conv_block(p6, 512)                # (4, 4, 512)

        # Decoder
        d1 = self.decoder_block(b, x6, 512)         # (8, 8, 512)
        d2 = self.decoder_block(d1, x5, 512)        # (16, 16, 512)
        d3 = self.decoder_block(d2, x4, 512)        # (32, 32, 256)
        d4 = self.decoder_block(d3, x3, 256)        # (64, 64, 128)
        d5 = self.decoder_block(d4, x2, 256)        # (128, 128, 64)
        d6 = self.decoder_block(d5, x1, 256)        # (256, 256, 64)

        self.model = keras.Model(inputs, d6, name="FeatureExtractorNetwork")

    def conv_block(self, inputs, num_filters, batch_norm=True):
        x = layers.Conv2D(num_filters, 3, padding = "same",)(inputs)
        if(batch_norm):
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(num_filters, 3, padding = "same",)(x)
        if(batch_norm):
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        return x

    def encoder_block(self, inputs, num_filters):
        x = self.conv_block(inputs, num_filters)
        p = BlurPool2D()(x)
        p = layers.MaxPooling2D(pool_size = (2, 2))(x)
        return x, p

    def decoder_block(self, inputs, skip_features, num_filters):
        x = layers.UpSampling2D(interpolation = "bilinear")(inputs)
        x = layers.concatenate([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x


if __name__ == "__main__":
    outputs = FeatureExtractorNetwork().model.summary()




# Unet Architecture
# https://www.youtube.com/watch?v=NKOOA_xxCKE