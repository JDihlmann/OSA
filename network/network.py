import os
import sys
import math
import datetime
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from .featureExtractorNetwork import FeatureExtractorNetwork
from .geometryNetwork import GeomertyNetwork
from .loss import total_loss, geometry_loss, sdf_loss, color_loss, geometric_regularization, surface_projection_loss
from dataset.loader.utils.image import convert_points_to_image_coordinate

class Network:
    # End to end network combining the feature extractor network and the geometry network
    # Using a brige to get the positional information from the feature extractor network

    def __init__(self, image_dataset=None, position_dataset=None, image_input_shape=(512, 512, 3), position_input_shape=(256+3)):
        # networks 
        self.fNetwork = FeatureExtractorNetwork(image_input_shape)
        self.gNetwork = GeomertyNetwork(position_input_shape )

        # datasets
        self.image_dataset = image_dataset
        self.position_dataset = position_dataset

        # path 
        parentPath = str(Path(__file__).parent.parent.absolute())

        # tensorboard 
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = parentPath + '/train/logs/' + current_time + '/train'
        val_log_dir = parentPath + '/train/logs/' + current_time + '/val'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # model saving 
        self.checkpoint_path_fNetwork = "models/f/cp-{epoch:04d}.ckpt"
        self.checkpoint_dir_fNetwork = parentPath + "/train/" +  self.checkpoint_path_fNetwork
        self.checkpoint_path_gNetwork = "models/g/cp-{epoch:04d}.ckpt"
        self.checkpoint_dir_gNetwork = parentPath + "/train/" +  self.checkpoint_path_gNetwork

        # trainable variables
        self.k = tf.Variable(250., trainable=True) # different version init with high k to force labels


    def train(self, epoch_count=100):

        # learning rate decay
        lr_schedule = tf.keras.optimizers.schedules.LinearDecay(initial_learning_rate=1e-4, decay_steps=50000, decay_rate=0.9) # Original 50000


        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        for epoch in range(epoch_count):
            print("epoch: {}".format(epoch))
            
            # training
            tl = []
            train_steps_length = int(self.image_dataset.train_size /  self.image_dataset.batch_size)
            with tqdm(total=train_steps_length) as pbar:
                for step, (image_batch, avatar_name_batch) in enumerate(self.image_dataset.train_ds): 

                    # train step
                    loss, other_losses, tape = self.train_step(image_batch, avatar_name_batch, step_count=step+epoch*train_steps_length)
                    gl, sl, gr, cl, spl = other_losses

                    # losses 
                    tl.append(loss.numpy())
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss.numpy().item(), step=step + (epoch*train_steps_length))
                        tf.summary.scalar('geometry_loss', gl.numpy().item(), step=step + (epoch*train_steps_length))
                        tf.summary.scalar('sdf_loss', sl.numpy().item(), step=step + (epoch*train_steps_length))
                        tf.summary.scalar('geometric_regularization', gr.numpy().item(), step=step + (epoch*train_steps_length))
                        tf.summary.scalar('color_loss', cl.numpy().item(), step=step + (epoch*train_steps_length))
                        tf.summary.scalar('surface_projection_loss', spl.numpy().item(), step=step + (epoch*train_steps_length))

                    # gradients
                    gradients = tape.gradient(loss, self.gNetwork.model.trainable_variables+self.fNetwork.model.trainable_variables+[self.k])
                    optimizer.apply_gradients(zip(gradients, self.gNetwork.model.trainable_variables+self.fNetwork.model.trainable_variables+[self.k]))
                    pbar.update(1)
            
            print("mean train loss: {}".format(np.array(tl).mean()))
            print("k : {}".format(self.k.numpy()))
            print("lr: {:0.7f}".format(optimizer._decayed_lr(tf.float32).numpy()))

            # validation
            '''  vl, vl_gl, vl_sl, vl_gr, vl_cl, vl_spl = ([], [], [], [], [], [])
            val_steps_length = int(self.image_dataset.val_size / self.image_dataset.batch_size)
            with tqdm(total=val_steps_length) as pbar:
                for step, (image_batch, avatar_name_batch) in enumerate(self.image_dataset.val_ds): 

                    # validation step
                    loss, other_losses = self.validation_step(image_batch, avatar_name_batch, step_count=(epoch+1)*train_steps_length)
                    gl, sl, gr, cl, spl = other_losses

                    # losses 
                    vl.append(loss.numpy())
                    vl_gl.append(gl.numpy().item())
                    vl_sl.append(sl.numpy().item())
                    vl_gr.append(gr.numpy().item())
                    vl_cl.append(cl.numpy().item())
                    vl_spl.append(spl.numpy().item())
                    pbar.update(1)

            with self.val_summary_writer.as_default():
                tf.summary.scalar('loss', np.array(vl).mean(), step=(epoch+1)*train_steps_length)
                tf.summary.scalar('geometry_loss', np.array(vl_gl).mean(), step=(epoch+1)*train_steps_length)
                tf.summary.scalar('sdf_loss', np.array(vl_sl).mean(), step=(epoch+1)*train_steps_length)
                tf.summary.scalar('geometric_regularization', np.array(vl_gr).mean(), step=(epoch+1)*train_steps_length)
                tf.summary.scalar('color_loss', np.array(vl_cl).mean(), step=(epoch+1)*train_steps_length)
                tf.summary.scalar('surface_projection_loss', np.array(vl_spl).mean(), step=(epoch+1)*train_steps_length)
                tf.summary.scalar('k', self.k.numpy().item(), step=(epoch+1)*train_steps_length)
            
            print("mean validation loss: {}".format(np.array(vl).mean()))  '''

            # save model
            self.fNetwork.model.save_weights(self.checkpoint_path_fNetwork.format(epoch=epoch))
            self.gNetwork.model.save_weights(self.checkpoint_path_gNetwork.format(epoch=epoch))
            

    def train_step(self, image_batch, avatar_name_batch, step_count=0):

        with tf.GradientTape() as tape:
            # feature extractor network
            feature_voxel_grid_batch = self.fNetwork.model(image_batch, training=True)

            # construct geometry network input
            inputs_gnetwork_near, true_points_near, true_colors_near, true_normals_near, true_distances_near = self.construct_input(feature_voxel_grid_batch, avatar_name_batch, "near")
            inputs_gnetwork_far, true_points_far, true_colors_far, true_normals_far, true_distances_far = self.construct_input(feature_voxel_grid_batch, avatar_name_batch, "far")
            inputs_gnetwork_near = tf.reshape(inputs_gnetwork_near, (self.image_dataset.batch_size, self.position_dataset.batch_size, inputs_gnetwork_near.shape[1]))
            inputs_gnetwork_far = tf.reshape(inputs_gnetwork_far, (self.image_dataset.batch_size, self.position_dataset.batch_size, inputs_gnetwork_far.shape[1]))
            inputs_gnetwork = tf.concat([inputs_gnetwork_near, inputs_gnetwork_far], axis=1)
            inputs_gnetwork = tf.reshape(inputs_gnetwork, (self.image_dataset.batch_size*self.position_dataset.batch_size*2, inputs_gnetwork.shape[2]))

            # geometry network
            output = self.gNetwork.model(inputs_gnetwork, training=True)
            estimated_distances_near, estimated_distances_far, estimated_normals_near, estimated_normals_far, estimated_colors_near, estimated_colors_far = self.restructure_output(output)

            # loss 
            gl = geometry_loss(estimated_distances_near, true_normals_near, estimated_normals_near, step_count)
            sl = sdf_loss(true_distances_far, estimated_distances_far, self.k) 
            gr = geometric_regularization(estimated_normals_far)
            cl = color_loss(true_colors_near, estimated_colors_near, true_colors_far, estimated_colors_far)
            spl = surface_projection_loss(true_points_far, true_distances_far, estimated_distances_far, true_normals_far, estimated_normals_far)

            tl = total_loss(gl, sl, gr, cl, surface_projection_loss=spl)

            # check if the loss is nan
            if np.isnan(tl.numpy()):
                print("Loss is nan")
                exit()
                
        return tl, (gl, sl, gr, cl, spl), tape
    

    def validation_step(self, image_batch, avatar_name_batch, step_count=0):
        feature_voxel_grid_batch = self.fNetwork.model.predict(image_batch, verbose=0)
            
        # construct geometry network input
        inputs_gnetwork_near, true_points_near, true_colors_near, true_normals_near, true_distances_near = self.construct_input(feature_voxel_grid_batch, avatar_name_batch, "near")
        inputs_gnetwork_far, true_points_far, true_colors_far, true_normals_far, true_distances_far = self.construct_input(feature_voxel_grid_batch, avatar_name_batch, "far")
        inputs_gnetwork_near = tf.reshape(inputs_gnetwork_near, (self.image_dataset.batch_size, self.position_dataset.batch_size, inputs_gnetwork_near.shape[1]))
        inputs_gnetwork_far = tf.reshape(inputs_gnetwork_far, (self.image_dataset.batch_size, self.position_dataset.batch_size, inputs_gnetwork_far.shape[1]))
        inputs_gnetwork = tf.concat([inputs_gnetwork_near, inputs_gnetwork_far], axis=1)
        inputs_gnetwork = tf.reshape(inputs_gnetwork, (self.image_dataset.batch_size*self.position_dataset.batch_size*2, inputs_gnetwork.shape[1]))
        
        # geometry network
        output = self.gNetwork.model.predict(inputs_gnetwork, verbose=0)
        estimated_distances_near, estimated_distances_far, estimated_normals_near, estimated_normals_far, estimated_colors_near, estimated_colors_far = self.restructure_output(output)

        # loss 
        gl = geometry_loss(estimated_distances_near, true_normals_near, estimated_normals_near, step_count)
        sl = sdf_loss(true_distances_far, estimated_distances_far, self.k) 
        gr = geometric_regularization(estimated_normals_far)
        cl = color_loss(true_colors_near, estimated_colors_near, true_colors_far, estimated_colors_far)
        spl = surface_projection_loss(true_points_far, true_distances_far, estimated_distances_far, true_normals_far, estimated_normals_far)

        tl = total_loss(gl, sl, gr, cl, surface_projection_loss=spl)

        return tl, (gl, sl, gr, cl, spl)

       
    def construct_input(self, feature_voxel_grid_batch, avatar_name_batch, distance_type):
        points_batch, colors_batch, normals_batch, distances_batch, inputs_batch = ([], [], [], [], [])
        names = avatar_name_batch.numpy()[:, 1]

        # loop over each image in the batch
        for name_idx in range(len(names)):
            name = names[name_idx].decode('utf-8')

            # sample points 
            ds = self.position_dataset.avatar_ds[distance_type][name]
            positions_batch = ds.take(1)

            # query feature voxel grid for each point
            for idx_points, points, distances, colors, normals in positions_batch:
                voxel_slice = tf.gather(feature_voxel_grid_batch, tf.constant([name_idx]), axis=0)
                voxel_slice = tf.reshape(voxel_slice, (voxel_slice.shape[1], voxel_slice.shape[2], voxel_slice.shape[3]))
                voxel_slice = tf.gather_nd(voxel_slice, [idx_points])
                voxel_slice = tf.reshape(voxel_slice, (idx_points.shape[0], voxel_slice.shape[2]))

                # append to batch
                colors_batch.append(colors)
                normals_batch.append(normals)
                distances_batch.append(distances)
                points_batch.append(points)
                
                # append points
                voxel_points_slice = tf.concat([points, voxel_slice], axis=1)
                inputs_batch.append(voxel_points_slice)
                    
        inputs_batch = tf.concat(inputs_batch, 0)
        colors_batch = tf.stack(colors_batch, 0)
        normals_batch = tf.stack(normals_batch, 0)
        distances_batch = tf.stack(distances_batch, 0)
        points_batch = tf.stack(points_batch, 0)

        return inputs_batch, points_batch, colors_batch, normals_batch, distances_batch
    
    def restructure_output(self, output):
        output = tf.reshape(output, (self.image_dataset.batch_size, self.position_dataset.batch_size * 2, output.shape[1]))
            
        # split depth and color from output
        estimated_distances = tf.gather(output, tf.constant([0]), axis=2)
        estimated_colors = tf.gather(output, tf.constant([1, 2, 3]), axis=2)
        estimated_normals = tf.gather(output, tf.constant([4, 5, 6]), axis=2)

        # split estimated into near and far 
        estimated_distances_near = estimated_distances[:, 0:self.position_dataset.batch_size]
        estimated_distances_far = estimated_distances[:, self.position_dataset.batch_size:]
        estimated_normals_near = estimated_normals[:, 0:self.position_dataset.batch_size, :]
        estimated_normals_far = estimated_normals[:, self.position_dataset.batch_size:, :]
        estimated_colors_near = estimated_colors[:, 0:self.position_dataset.batch_size, :]
        estimated_colors_far = estimated_colors[:, self.position_dataset.batch_size:, :]
        return estimated_distances_near, estimated_distances_far, estimated_normals_near, estimated_normals_far, estimated_colors_near, estimated_colors_far

    
    def load_weights(self, epoch):
        self.fNetwork.model.load_weights(self.checkpoint_dir_fNetwork.format(epoch=epoch))
        self.gNetwork.model.load_weights(self.checkpoint_dir_gNetwork.format(epoch=epoch))

    def inference(self, image, points):
        # prepare image
        image_tensor = tf.convert_to_tensor(image, dtype=tf.int32)
        image_tensor = tf.reshape(image_tensor, (1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]))

        # run feature extractor model
        feature_voxel_grid = self.fNetwork.model.predict(image_tensor)
        feature_voxel_grid  = tf.reshape(feature_voxel_grid, (feature_voxel_grid.shape[1], feature_voxel_grid.shape[2], feature_voxel_grid.shape[3]))

        # prepare points
        idx_points = convert_points_to_image_coordinate(points, image.shape[0])
        voxel_slices = tf.gather_nd(feature_voxel_grid, [idx_points])
        points_tensor  = tf.convert_to_tensor(points, dtype=tf.float32)
        points_tensor = tf.reshape(points_tensor, (1, points_tensor.shape[0], points_tensor.shape[1]))
        voxel_points_tensor = tf.concat([points_tensor, voxel_slices], 2)
        voxel_points_tensor = tf.reshape(voxel_points_tensor, (voxel_points_tensor.shape[1], voxel_points_tensor.shape[2]))

        print("voxel_points_tensor shape: ", voxel_points_tensor.shape)

        # run geometry network
        output = self.gNetwork.model.predict(voxel_points_tensor)

        distances = output[:, 0]
        colors = output[:, 1:4]
        normals = output[:, 4:]

        return distances, colors, normals
    