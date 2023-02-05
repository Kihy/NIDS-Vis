import tensorflow as tf
from tensorflow.keras import layers, losses
# import tensorflow_addons as tfa
from helper import *
from abc import ABC, abstractmethod
from ripser import ripser
from gudhi.tensorflow import RipsLayer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.backend.set_floatx("float32")


def get_ae(ae_type, ae_param):
    if ae_type == "LSTM":
        ae = LSTM_AE(**ae_param)

    elif ae_type == "FC":
        ae = FC_AE(**ae_param)

    elif ae_type == "CONV":
        ae = Conv_AE(**ae_param)
    else:
        raise ValueError(f"No model found with {ae_type}")
    ae.compute_output_shape(input_shape=(None, 100))
    return ae


class Autoencoder(tf.keras.Model, ABC):
    def __init__(self, shape, input_dim, latent_dim, num_neurons, denoise, double_recon,
                 latent_slices, scaler, window_size, step_size, fc, batch_size, reduce,
                 alpha, include_losses, opt,*args, **kwargs):
        super(Autoencoder, self).__init__()
        self.nb_epochs = tf.Variable(0., trainable=False)
        self.scaler = scaler
        self.fc = fc
        self.latent_dim = latent_dim
        self.slices = latent_slices
        self.shape = shape
        self.reduce = reduce
        self.alpha = alpha
        self.input_dim = input_dim
        self.window_size = window_size
        self.step_size = step_size
        self.denoise = denoise
        self.include_losses = include_losses + ["total_loss"]
        self.encoder, self.decoder = self.create_encoder_decoder(num_neurons)
        if "topological_loss" in include_losses:
            self.rl = RipsLayer(homology_dimensions=[0])

        self.losses_list = {name: tf.keras.metrics.Mean()
                            for name in self.include_losses}
        self.batch_size = batch_size
        self.optimizer = opt

        self.double_recon = double_recon

    def get_config(self):
        config = super(Autoencoder, self).get_config()
        config.update({"fc": self.fc,
                       "latent_dim": self.latent_dim,
                       "latent_slices": self.slices,
                       "shape": self.shape,
                       "input_dim": self.input_dim,
                       "window_size": self.window_size,
                       "step_size": self.step_size,
                       "encoder": self.encoder,
                       "decoder": self.decoder,
                       "batch_size": self.batch_size,
                       "custom_train_step": self.custom_train_step,
                       "increment_epoch": self.increment_epoch,
                       "nb_epochs": self.nb_epochs.numpy(),
                       "losses_list": self.losses_list,
                       "reduce": self.reduce,
                       "scaler": self.scaler,
                       "alpha": self.alpha,
                       "include_losses": self.include_losses,
                       "denoise": self.denoise,

                       "double_recon": self.double_recon,
                       "optimizer": self.optimizer})
        return config

    @ abstractmethod
    def create_encoder_decoder(self):
        pass

    @ abstractmethod
    def preprocess(self, x):
        pass

    @ abstractmethod
    def postprocess(self, x):
        pass

    def compute_output_shape(self, input_shape):
        return [(None, 2), (None, 1), (None, 100)]

    # @ tf.function
    def call(self, x, training=False, denoise=False, double_recon=False):

        x = self.scaler.transform(x)
        if denoise:
            noise = tf.random.uniform(
                shape=x.shape, minval=-0.05, maxval=0.05, dtype=x.dtype)
            input_x = x + x * noise
        else:
            input_x = x

        # encode_x = x
        encode_x = self.preprocess(input_x)
        encode_x = self.encoder(encode_x, training=training)

        decoded_x = self.decoder(encode_x, training=training)
        decoded_x1 = self.postprocess(decoded_x)
        inverse_x = self.scaler.inverse_transform(decoded_x1)
        recon_loss = tf.keras.losses.mse(x, decoded_x1)
        if double_recon:
            re_encoded_x = self.encoder(decoded_x, training=training)
            recon_loss += tf.keras.losses.mse(encode_x, re_encoded_x)
        return encode_x, recon_loss, inverse_x

    def get_regularization_loss(self):
        return tf.reduce_sum(self.losses)

    @tf.function
    def custom_train_step(self, x, loss_norm=True):

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)

            encoded, recon_loss, decoded_x = self.call(
                x, training=True, denoise=self.denoise, double_recon=self.double_recon)

            # get existing losses (activity regularization)
            total_loss = 0.
            # tf.print(loss)
            for loss_name in self.include_losses:

                if loss_name == "recon_loss":

                    if loss_norm:
                        reduced_recon_loss = self.reduce(recon_loss)
                    else:
                        reduced_recon_loss = self.reduce(
                            tf.keras.losses.mse(x, decoded_x))
                    loss_value = reduced_recon_loss

                elif loss_name == "contractive_loss":
                    jacobian = tape.batch_jacobian(encoded, x)
                    loss_value = tf.reduce_mean(
                        tf.norm(jacobian, axis=-1))

                elif loss_name == "entropy":
                    drecon_dx = tape.gradient(reduced_recon_loss, x)
                    pos_weight = tf.math.count_nonzero(
                        (tf.math.greater_equal(drecon_dx, 0)), axis=0)
                    pos_prob = pos_weight / drecon_dx.shape[0]
                    neg_prob = 1.0 - pos_prob
                    entropy = -(tf.math.multiply_no_nan(tf.math.log(pos_prob), pos_prob)
                                + tf.math.multiply_no_nan(tf.math.log(neg_prob), neg_prob))
                    entropy = -tf.math.top_k(-entropy, k=10).values
                    loss_value = tf.cast(tf.reduce_mean(entropy), tf.float32)

                elif loss_name == "dist_loss":
                    x_scaled = self.scaler.transform(x)
                    lat_distance = pairwise_distance(encoded)
                    x_distance = pairwise_distance(x_scaled)
                    loss_value = tf.reduce_mean(
                        tf.keras.losses.mse(lat_distance, x_distance))

                elif loss_name == "dist_loss2":
                    x_scaled = self.scaler.transform(x)
                    lat_distance = pairwise_distance(encoded)
                    x_distance = pairwise_distance(x_scaled)

                    normalised_lat = lat_distance / \
                        tf.reduce_max(lat_distance, axis=1, keepdims=True)
                    normalised_x = x_distance / \
                        tf.reduce_max(x_distance, axis=1, keepdims=True)

                    distance1 = normalised_lat[:, None,
                                               :] - normalised_lat[:, :, None]
                    x1 = normalised_x[:, None, :] - normalised_x[:, :, None]

                    loss_value = tf.reduce_mean(tf.reduce_sum(
                        tf.nn.relu(-distance1 * x1), axis=[1, 2]))

                elif loss_name == "sw_loss":
                    sw_loss = self.calc_sw_loss(encoded)
                    loss_value = self.alpha * sw_loss

                elif loss_name == "ranking_loss":
                    distance = tf.linalg.norm(encoded, axis=1)
                    distance1 = distance[None, :] - distance[:, None]
                    recon1 = recon_loss[None, :] - recon_loss[:, None]

                    dist_loss = tf.nn.relu(-distance1 * recon1)

                    dist_loss = tf.reduce_sum(dist_loss, axis=1)
                    loss_value = tf.reduce_mean(dist_loss)
                elif loss_name == "sliced_topo_loss":
                    x_scaled = self.scaler.transform(x)
                    slices = generate_theta(
                        self.slices, self.input_dim, dtype=encoded.dtype)
                    encoded_slice, _, _ = self.call(slices)
                    features_dist = point_line_distance(
                        slices, x_scaled)
                    features_dist /= tf.reduce_max(features_dist,
                                                   axis=1, keepdims=True)

                    latent_dist = point_line_distance(
                        encoded_slice, encoded)
                    latent_dist /= tf.reduce_max(latent_dist,
                                                 axis=1, keepdims=True)

                    loss_value = tf.losses.mae(features_dist, latent_dist)

                elif loss_name == "topological_loss":
                    x_scaled = self.scaler.transform(x)
                    x_dist = pairwise_distance(x_scaled)
                    latent_dist = pairwise_distance(encoded)
                    # x_sig = ripser(x_scaled, maxdim=0)['dgms'][0][:-1]
                    # latent_sig = ripser(encoded, maxdim=0)['dgms'][0][:-1]

                    x_sig = self.rl(x_scaled)[0][0]
                    latent_sig = self.rl(encoded)[0][0]
                    x_pairs = self.get_edge_idx(x_dist, x_sig[:, 1])
                    latent_pairs = self.get_edge_idx(
                        latent_dist, latent_sig[:, 1])

                    loss_value = tf.cast(
                        sig_error(tf.gather_nd(latent_dist, x_pairs), tf.gather_nd(x_dist, x_pairs)) +
                        sig_error(tf.gather_nd(latent_dist, latent_pairs), tf.gather_nd(x_dist, latent_pairs)), dtype="float32")

                total_loss += loss_value
                self.losses_list[loss_name].update_state(loss_value)

    # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

    # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.losses_list["total_loss"].update_state(total_loss)

        loss_results = {}
        for loss_name, loss in self.losses_list.items():
            loss_results[loss_name] = loss.result()
        return loss_results

    @tf.function
    def get_edge_idx(self, distance, edge_weights):
        pairs1 = tf.where(tf.abs(edge_weights[tf.newaxis, tf.newaxis, :] -
                          tf.linalg.band_part(distance, 0, -1)[:, :, tf.newaxis]) < 1e-6)
        order = tf.argsort(pairs1[:, -1], axis=0)
        return tf.gather(pairs1, order)[:, :-1]

    @ property
    def metrics(self):
        return self.losses_list

    @ tf.function
    def increment_epoch(self):
        self.nb_epochs.assign_add(1.)
        self.reset_metrics()
        # return self.nb_epochs
        return tf.cast(self.nb_epochs, dtype=tf.int64)

    def reset_metrics(self):
        for loss in self.losses_list.values():
            loss.reset_state()

    @tf.function
    def calc_sw_loss(self, encoded):
        # Define a Keras Variable for \theta_ls
        theta = generate_theta(
            self.slices, self.latent_dim, dtype=encoded.dtype)

        # Define a Keras Variable for samples of z
        z = generate_z(encoded.shape[0], self.latent_dim,
                       self.shape, dtype=encoded.dtype, center=0)

        # Let projae be the projection of the encoded samples
        projae = tf.tensordot(encoded, tf.transpose(theta), axes=1)
        # projae += tf.expand_dims(tf.norm(encoded, axis=1), axis=1)
        # Let projz be the projection of the $q_Z$ samples
        projz = tf.tensordot(z, tf.transpose(theta), axes=1)
        # projz += tf.expand_dims(tf.norm(z, axis=1), axis=1)
        # Calculate the Sliced Wasserstein distance by sorting
        # the projections and calculating the L2 distance between
        sw_loss = self.reduce((tf.sort(tf.transpose(projae))
                               - tf.sort(tf.transpose(projz)))**2)
        return sw_loss


class LSTM_AE(Autoencoder):
    def create_encoder_decoder(self):
        encoder = tf.keras.Sequential([

            layers.LSTM(32, return_sequences=False),
            layers.Dense(self.inter_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(self.latent_dim)
        ])
        decoder = tf.keras.Sequential([
            layers.Dense(self.inter_dim, activation="relu",
                         input_shape=(self.latent_dim,)),
            layers.RepeatVector(self.window_size,),
            layers.LSTM(32, return_sequences=True),
            layers.TimeDistributed(layers.Dense(100))
        ])

        return encoder, decoder

    @ tf.function
    def preprocess(self, x):
        x = cluster_feature(x, self.fc)
        return generate_sliding_window(x, self.window_size, self.step_size)

    def postprocess(self, x):
        x = unslide_window(x, self.window_size,
                           self.step_size, self.batch_size)
        return uncluster_feature(x, self.fc)


class TRec(tf.keras.layers.Layer):
    def __init__(self, theta, **kwargs):
        super(TRec, self).__init__(**kwargs)
        self.theta = tf.constant(theta, dtype="float64")

    def call(self, inputs, training=False):

        return tf.where(inputs < self.theta, 1e-3 * inputs, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "theta": self.theta.numpy(),
        })
        return config


class FC_AE(Autoencoder):
    def create_encoder_decoder(self, num_neurons):
        use_bias = True
        encoder_layers = [layers.Dense(num_neurons[0], input_shape=(self.input_dim,), use_bias=use_bias),
                          # tf.keras.layers.BatchNormalization(),
                          layers.LeakyReLU(0.2),
                          ]
        for i in num_neurons[1:]:
            encoder_layers.extend([layers.Dense(i, use_bias=use_bias),
                                   # tf.keras.layers.BatchNormalization(),
                                   layers.LeakyReLU(0.2),
                                   ])
        encoder_layers.append(layers.Dense(self.latent_dim, use_bias=use_bias))
        encoder = tf.keras.Sequential(encoder_layers)

        decoder_layers = [layers.Dense(num_neurons[-1], input_shape=(self.latent_dim,), use_bias=use_bias),
                          # tf.keras.layers.BatchNormalization(),
                          layers.LeakyReLU(0.2)]
        for i in num_neurons[::-1][1:]:
            decoder_layers.extend([layers.Dense(i, use_bias=use_bias),
                                   # tf.keras.layers.BatchNormalization(),
                                   layers.LeakyReLU(0.2),
                                   ])
        decoder_layers.append(layers.Dense(self.input_dim, use_bias=use_bias))
        decoder = tf.keras.Sequential(decoder_layers)

        return encoder, decoder

    @ tf.function
    def preprocess(self, x):
        x = cluster_feature(x, self.fc)
        return x

    @ tf.function
    def postprocess(self, x):
        return uncluster_feature(x, self.fc)


class Conv_AE(Autoencoder):
    def create_encoder_decoder(self):
        encoder = tf.keras.Sequential([
            layers.Conv1D(8, 10, padding='same',
                          input_shape=(self.window_size, 100)),
            layers.LeakyReLU(alpha=0.2),
            layers.AveragePooling1D(4, padding='same'),
            layers.Conv1D(16, 10, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.AveragePooling1D(4, padding='same'),
            layers.Conv1D(24, 10, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.AveragePooling1D(4, padding='same'),
            layers.Flatten(),
            layers.Dense(self.inter_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(self.latent_dim)
        ])

        decoder = tf.keras.Sequential([
            layers.Dense(self.inter_dim, input_shape=(
                self.latent_dim,)),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(24, activation='relu'),
            layers.Reshape((1, 24)),
            layers.UpSampling1D(4),
            layers.Conv1DTranspose(24, 10, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.UpSampling1D(4),
            layers.Conv1DTranspose(16, 10, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.UpSampling1D(4),
            layers.Conv1DTranspose(8, 10, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(
                100, 10, padding='same', activation='relu')
        ])
        # print(encoder.summary())
        # print(decoder.summary())
        return encoder, decoder

    @ tf.function
    def preprocess(self, x):
        x = cluster_feature(x, self.fc)
        return generate_sliding_window(x, self.window_size, self.step_size)

    @ tf.function
    def postprocess(self, x):
        x = unslide_window(x, self.window_size,
                           self.step_size, self.batch_size)
        return uncluster_feature(x, self.fc)
