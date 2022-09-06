import os
import math
import datetime
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from .featureExtractorNetwork import FeatureExtractorNetwork
from .geometryNetwork import GeomertyNetwork
from .lookupNetwork import LookupNetwork
from.loss import total_loss, geometry_loss, sdf_loss, color_loss, geometric_regularization, surface_projection_loss
from .datasetLoader.helper.image import convert_points_to_image_coordinate

class Network:
    # End to end network combining the feature extractor network and the geometry network
    # Using a brige to get the positional information from the feature extractor network

    def __init__(self, image_dataset=None, position_dataset=None, image_input_shape=(512, 512, 3), position_input_shape=(256+3)):
        # networks 
        self.fNetwork = FeatureExtractorNetwork(image_input_shape)
        self.gNetwork = GeomertyNetwork(position_input_shape )
        self.lNetwork = LookupNetwork(position_input_shape)

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
        self.k = tf.Variable(250., trainable=True) # different version init with high k to force labels

        # empty grid tensor
        self.grid = self.positional_lookup_table(dim=(512, 512, 1))


    def train(self, epoch_count=100):

        # learning rate decay
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=20000, decay_rate=0.9) # Original 50000


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

            # save model
            self.fNetwork.model.save_weights(self.checkpoint_path_fNetwork.format(epoch=epoch))
            self.gNetwork.model.save_weights(self.checkpoint_path_gNetwork.format(epoch=epoch))
            


    def train_step(self, image_batch, avatar_name_batch, step_count=0):

        with tf.GradientTape() as tape:
            # feature extractor network
            latent_space_batch, feature_voxel_grid_batch = self.fNetwork.model(image_batch, training=True)

            # construct geometry network input
            inputs_lnetwork_near, true_points_near, true_colors_near, true_normals_near, true_distances_near = self.construct_linput(latent_space_batch, avatar_name_batch, "near")
            inputs_lnetwork_far, true_points_far, true_colors_far, true_normals_far, true_distances_far = self.construct_linput(latent_space_batch, avatar_name_batch, "far")


            inputs_lnetwork_near = tf.reshape(inputs_lnetwork_near, (self.image_dataset.batch_size, self.position_dataset.batch_size, inputs_lnetwork_near.shape[1]))
            inputs_lnetwork_far = tf.reshape(inputs_lnetwork_far, (self.image_dataset.batch_size, self.position_dataset.batch_size, inputs_lnetwork_far.shape[1]))
            inputs_lnetwork = tf.concat([inputs_lnetwork_near, inputs_lnetwork_far], axis=1)
            inputs_lnetwork = tf.reshape(inputs_lnetwork, (self.image_dataset.batch_size*self.position_dataset.batch_size*2, inputs_lnetwork.shape[2]))

            # lookup network
            scalar, mu, sigma = self.lNetwork.model(inputs_lnetwork, training=True)
            inputs_gnetwork = self.construct_ginput(feature_voxel_grid_batch, inputs_lnetwork, scalar, mu, sigma)

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

    def construct_ginput(self, feature_voxel_grid_batch, inputs_lnetwork, scalar, mu, sigma):
        inputs_batch = []

        # loop over batch
        for i in range(inputs_lnetwork.shape[0]): 
            feature_voxel_grid_batch_index = i // (self.position_dataset.batch_size * 2)
            point = inputs_lnetwork[i,0:3]
            point_mu = mu[i]
            point_sigma = sigma[i] + + 1e-8
            point_scalar = scalar[i]
            mnfc = tfd.MultivariateNormalDiag(point_mu, point_sigma) 
            grid = mnfc.prob(self.grid) * point_scalar
            grid = tf.reshape(grid, [grid.shape[0], grid.shape[1], 1])

            feature_voxel_slice =  feature_voxel_grid_batch[feature_voxel_grid_batch_index]
            point_feature = tf.multiply(grid, feature_voxel_slice)
            point_feature = tf.reduce_sum(tf.reduce_sum(point_feature, axis=0), axis=0)
            point_and_feature = tf.concat([point, point_feature], axis=0)
            inputs_batch.append(point_and_feature)

        return tf.stack(inputs_batch, 0)
       
    def construct_linput(self, latent_space_batch, avatar_name_batch, distance_type):
        points_batch, colors_batch, normals_batch, distances_batch, inputs_batch = ([], [], [], [], [])
        names = avatar_name_batch.numpy()[:, 1]

        # loop over each image in the batch
        for name_idx in range(len(names)):
            name = names[name_idx].decode('utf-8')

            # sample points 
            ds = self.position_dataset.avatar_ds[distance_type][name]
            positions_batch = ds.take(1)

            # latent space
            latent_space = latent_space_batch[name_idx]
            latent_space = tf.repeat(tf.reshape(latent_space, [1, latent_space.shape[0]]), self.position_dataset.batch_size, axis=0)

            # query feature voxel grid for each point
            for idx_points, points, distances, colors, normals in positions_batch:
                # append to batch
                colors_batch.append(colors)
                normals_batch.append(normals)
                distances_batch.append(distances)
                points_batch.append(points)
                
                # append points
                points_and_latent_space = tf.concat([points, latent_space], axis=1)
                inputs_batch.append(points_and_latent_space)
                    
        inputs_batch = tf.concat(inputs_batch, 0)
        colors_batch = tf.stack(colors_batch, 0)
        normals_batch = tf.stack(normals_batch, 0)
        distances_batch = tf.stack(distances_batch, 0)
        points_batch = tf.stack(points_batch, 0)

        return inputs_batch, points_batch, colors_batch, normals_batch, distances_batch

    def positional_lookup_table(self, dim=(512,512,2)):
        # linspace tensor for x,y,z
        x = tf.linspace(-1, 1, dim[0])
        y = tf.linspace(-1, 1, dim[1])
        x, y = tf.meshgrid(x, y)
        xy = tf.stack([x, y], axis=2)
        xy = tf.cast(xy, tf.float32)
        return xy
    
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
        self.fNetwork.model.load_weights(self.checkpoint_path_fNetwork.format(epoch=epoch))
        self.gNetwork.model.load_weights(self.checkpoint_path_gNetwork.format(epoch=epoch))

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
    