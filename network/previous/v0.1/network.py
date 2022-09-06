import os
import datetime
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .featureExtractorNetwork import FeatureExtractorNetwork
from .geometryNetwork import GeomertyNetwork
from.loss import total_loss, geometry_loss, sdf_loss, color_loss, geometric_regularization

class Network:
    # End to end network combining the feature extractor network and the geometry network
    # Using a brige to get the positional information from the feature extractor network

    def __init__(self, image_dataset, position_dataset, image_input_shape=(512, 512, 3), position_input_shape=(256+3)):
        # networks 
        self.fNetwork = FeatureExtractorNetwork(image_input_shape)
        self.gNetwork = GeomertyNetwork(position_input_shape )

        # datasets
        self.image_dataset = image_dataset
        self.position_dataset = position_dataset

        # tensoorboard 
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/train'
        val_log_dir = 'logs/' + current_time + '/val'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # model saving 
        self.checkpoint_path_fNetwork = "models/f/cp-{epoch:04d}.ckpt"
        self.checkpoint_dir_fNetwork = os.path.dirname(self.checkpoint_path_fNetwork)
        self.checkpoint_path_gNetwork = "models/g/cp-{epoch:04d}.ckpt"
        self.checkpoint_dir_gNetwork = os.path.dirname(self.checkpoint_path_gNetwork)

        # trainable variables
        self.k = tf.Variable(100., trainable=True) # different version init with high k to force labels


    def train(self, epoch_count=100):

        # learning rate decay
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=50000, decay_rate=0.9)

        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        for epoch in range(epoch_count):
            print("Epoch: {}, ".format(epoch))
            
            # train step
            tl = []
            steps_length = int(self.image_dataset.train_size /  self.image_dataset.batch_size)
            with tqdm(total=steps_length) as pbar:
                for step, (image_batch, avatar_name_batch) in enumerate(self.image_dataset.train_ds): 

                    # train step
                    loss, other_losses, tape = self.train_step(image_batch, avatar_name_batch)
                    gl, sl, gr, cl = other_losses

                    # losses 
                    tl.append(loss.numpy())
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss.numpy().item(), step=step + (epoch*steps_length))
                        tf.summary.scalar('geometry_loss', gl.numpy().item(), step=step + (epoch*steps_length))
                        tf.summary.scalar('sdf_loss', sl.numpy().item(), step=step + (epoch*steps_length))
                        tf.summary.scalar('geometric_regularization', gr.numpy().item(), step=step + (epoch*steps_length))
                        tf.summary.scalar('color_loss', cl.numpy().item(), step=step + (epoch*steps_length))

                    # gradients
                    gradients = tape.gradient(target=loss,  sources= self.gNetwork.model.trainable_variables+self.fNetwork.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.gNetwork.model.trainable_variables+self.fNetwork.model.trainable_variables))
                    pbar.update(1)
            
            print("Loss: {}".format(np.array(tl).mean()))

            # validation step
            '''  vl = []
            steps_length = int(self.image_dataset.val_size / self.image_dataset.batch_size)
            with tqdm(total=steps_length) as pbar:
                for step, (image_batch, avatar_name_batch) in enumerate(self.image_dataset.val_ds): 

                    # train step
                    loss, other_losses, tape = self.train_step(image_batch, avatar_name_batch)
                    gl, sl, gr, cl = other_losses

                    # losses 
                    vl.append(loss.numpy())
                    with self.val_summary_writer.as_default():
                        tf.summary.scalar('loss', loss.numpy().item(), step=step + (epoch*steps_length))
                        tf.summary.scalar('geometry_loss', gl.numpy().item(), step=step + (epoch*steps_length))
                        tf.summary.scalar('sdf_loss', sl.numpy().item(), step=step + (epoch*steps_length))
                        tf.summary.scalar('geometric_regularization', gr.numpy().item(), step=step + (epoch*steps_length))
                        tf.summary.scalar('color_loss', cl.numpy().item(), step=step + (epoch*steps_length))

                    pbar.update(1)
            
            print("Loss: {}".format(np.array(vl).mean())) '''


            # save model
            self.fNetwork.model.save_weights(self.checkpoint_path_fNetwork.format(epoch=(epoch+1)))
            self.gNetwork.model.save_weights(self.checkpoint_path_gNetwork.format(epoch=(epoch+1)))



    def train_step(self, image_batch, avatar_name_batch):

        with tf.GradientTape() as tape:
            # feature extractor network
            feature_voxel_grid_batch = self.fNetwork.model(image_batch, training=True)

            # construct geometry network input
            inputs_gnetwork, true_points, true_colors, true_normals, true_depths, true_labels = self.construct_input_and_labels(feature_voxel_grid_batch, avatar_name_batch)

            # geometry network
            output = self.gNetwork.model(inputs_gnetwork, training=True)
            output = tf.reshape(output, (self.image_dataset.batch_size, self.position_dataset.batch_size, output.shape[1]))
            
            # split depth and color from output
            estimated_distance = tf.gather(output, tf.constant([0]), axis=2)
            estimated_colors = tf.gather(output, tf.constant([1, 2, 3]), axis=2)
            estimated_normals = tf.gather(output, tf.constant([4, 5, 6]), axis=2)

            # split estimated into near and far 
            estimated_distance_near = estimated_distance[:, 0:self.position_dataset.near_far_split_count]
            estimated_distance_far = estimated_distance[:, self.position_dataset.near_far_split_count:]
            estimated_normals_near = estimated_normals[:, 0:self.position_dataset.near_far_split_count, :]
            estimated_normals_far = estimated_normals[:, self.position_dataset.near_far_split_count:, :]
            estimated_colors_near = estimated_colors[:, 0:self.position_dataset.near_far_split_count, :]
            estimated_colors_far = estimated_colors[:, self.position_dataset.near_far_split_count:, :]

            # split true into near and far 
            true_normals_near = true_normals[:, 0:self.position_dataset.near_far_split_count, :]
            true_distance_far = true_depths[:, self.position_dataset.near_far_split_count:]
            true_colors_near = true_colors[:, 0:self.position_dataset.near_far_split_count, :]
            true_colors_far = true_colors[:, self.position_dataset.near_far_split_count:, :]

            # loss 
            gl = geometry_loss(estimated_distance_near, true_normals_near, estimated_normals_near)
            sl = sdf_loss(true_distance_far, estimated_distance_far, 100.0) # TODO: Add k back later 
            gr = geometric_regularization(estimated_normals_far)
            cl = color_loss(true_colors_near, estimated_colors_near, true_colors_far, estimated_colors_far)

            tl = total_loss(gl, sl, gr, cl)
            # check if the loss is nan
            if np.isnan(tl.numpy()):
                # save feature_voxel_grid_batch data
                np.savez("input_values.npz", voxel=feature_voxel_grid_batch.numpy(), image=image_batch.numpy(), avatar_name=avatar_name_batch.numpy(), true_points=true_points.numpy(), true_colors=true_colors.numpy(), true_normals=true_normals.numpy(), true_depths=true_depths.numpy(), true_labels=true_labels.numpy(), inputs_gnetwork=inputs_gnetwork.numpy(), output=output.numpy())

                print("Loss is nan")
                exit()
                

            
        return tl, (gl, sl, gr, cl), tape

       
    def construct_input_and_labels(self, feature_voxel_grid_batch, avatar_name_batch):
        points_batch = [] 
        colors_batch = []
        normals_batch = []
        depths_batch = []
        inputs_batch = []
        near_labels_batch = []

        names = avatar_name_batch.numpy()[:, 1]

        # loop over each image in the batch
        for name_idx in range(len(names)):
            name = names[name_idx].decode('utf-8')

            # sample points 
            ds = self.position_dataset.avatar_ds[name]
            positions_batch = ds.take(1)

            # query feature voxel grid for each point
            for idx_points, points, sdfs, colors, normals, near_labels in positions_batch:
                voxel_slice = tf.gather(feature_voxel_grid_batch, tf.constant([name_idx]), axis=0)
                voxel_slice = tf.reshape(voxel_slice, (voxel_slice.shape[1], voxel_slice.shape[2], voxel_slice.shape[3]))
                voxel_slice = tf.gather_nd(voxel_slice, [idx_points])
                voxel_slice = tf.reshape(voxel_slice, (idx_points.shape[0], voxel_slice.shape[2]))

                # append to batch
                colors_batch.append(colors)
                normals_batch.append(normals)
                depths_batch.append(sdfs)
                points_batch.append(points)
                near_labels_batch.append(near_labels)
                
                # append points
                # TODO: Encode points as fourier space 
                voxel_points_slice = tf.concat([points, voxel_slice], axis=1)
                inputs_batch.append(voxel_points_slice)
                    
        inputs_batch = tf.concat(inputs_batch, 0)
        colors_batch = tf.stack(colors_batch, 0)
        normals_batch = tf.stack(normals_batch, 0)
        depths_batch = tf.stack(depths_batch, 0)
        points_batch = tf.stack(points_batch, 0)
        near_labels_batch = tf.stack(near_labels_batch, 0)

        return inputs_batch, points_batch, colors_batch, normals_batch, depths_batch, near_labels_batch