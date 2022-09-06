import os
import json 
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

class ImageDatasetLoader:
    def __init__(self, batch_size=32, image_height=512, image_width=512):
        self.data_path = str(Path(__file__).parent.parent.absolute()) +   '/datasets/images/'
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width

        self.train_ds = None
        self.train_size = None

        self.val_ds = None
        self.val_size = None

    def load_data(self, validation_split=0.2, seed=42):
        # Load images and shuffle them
        list_ds = tf.data.Dataset.list_files(self.data_path + '*/*.png', shuffle=True, seed=seed)

        # Split into training and validation sets
        image_count = len(list_ds)
        val_size = int(image_count * validation_split)
        train_list_ds = list_ds.skip(val_size)
        val_list_ds = list_ds.take(val_size)

        # Save dataset split length
        self.train_size = len(train_list_ds)
        self.val_size = len(val_list_ds)

        # Load and preprocess images and labels
        self.train_ds = train_list_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        self.val_ds = val_list_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)

        # Configure for performance
        self.train_ds = self._configure_for_performance(self.train_ds)
        self.val_ds = self._configure_for_performance(self.val_ds)


    def _get_label(self, file_path):
        # Convert the path to a sting array of folder and image name
        label = tf.strings.split(file_path, os.path.sep)[-2:]
        avatar_name = tf.strings.split(label[1], ".")[0]
        label = tf.stack([label[0], avatar_name])
        return label

    def _get_img(self, file_path):
        # Convert the path to an image tensor
        image = tf.io.read_file(file_path)
        image = tf.io.decode_png(image, channels=3)

        image = tf.image.convert_image_dtype(image, tf.float32)
        # image = image / tf.constant(255, dtype=tf.float32)
        
        image = tf.image.resize(image, [self.image_height, self.image_width])
        return image 
    
    def _process_path(self, file_path):
        label = self._get_label(file_path)
        image = self._get_img(file_path)
        return image, label
    
    def _configure_for_performance(self, ds):
        # ds = ds.cache()
        ds = ds.shuffle(buffer_size=200)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def plot_batch(self, ds):
        image_batch, label_batch = next(iter(ds))

        plt.figure(figsize=(10 * (self.batch_size / 12), 10))
        for i in range(self.batch_size):
            plt.subplot(4, int(self.batch_size / 4), i + 1)
            plt.tight_layout()
            plt.imshow((image_batch[i][:, :, :] * 255).numpy().astype(int))
            plt.title(label_batch[i][1].numpy().decode('utf-8'))
            plt.axis('off')
        plt.show()

    def plot_image(self, env_name, avatar_name):
        image = self.load_image(env_name, avatar_name)

        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.axis('off')
        plt.title(avatar_name)
        plt.show()

    def load_image(self, env_name, avatar_name):
        path = self.data_path + env_name + '/' + avatar_name + '.png'
        image = plt.imread(path)
        return image

    def load_image_info(self, env_name, avatar_name):
        path = self.data_path + env_name + '/info.json'
        with open(path) as f:
            info = json.load(f)
            image_info = info[avatar_name]
        return image_info

    def pretty_print_image_info(self, env_name, avatar_name):
        image_info = self.load_image_info(env_name, avatar_name)
        
        # avatar name 
        print('Avatar Name: ', avatar_name)
        # environment name
        print('Environment Name: ', env_name)
        print('----------------------------------------------------------------------------------------------------------')
        # hdr name 
        print('HDR: ', image_info['hdr'])
        print('----------------------------------------------------------------------------------------------------------')
        # avatar position
        print('Avatar Position: ', image_info['avatar']['position'])
        # avatar rotation
        print('Avatar Rotation: ', image_info['avatar']['rotation'])
        # avatar bone
        print('Avatar Bone: ', image_info['avatar']['bone'])
        print('----------------------------------------------------------------------------------------------------------')
        # camera position
        print('Camera Position: ', image_info['camera']['position'])
        # camera target
        print('Camera Target: ', image_info['camera']['target'])


# Coded witht the help of following resources:
# https://www.tensorflow.org/tutorials/load_data/images