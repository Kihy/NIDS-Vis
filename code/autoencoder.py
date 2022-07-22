import tensorflow as tf
from tensorflow.keras import layers, losses

from helper import *
from abc import ABC, abstractmethod
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_ae(ae_type, shape, ae_param):
    if ae_type == "LSTM":
        ae = LSTM_AE(shape, **ae_param)

    elif ae_type == "FC":
        ae = FC_AE(shape, **ae_param)

    elif ae_type == "CONV":
        ae = Conv_AE(shape, **ae_param)
    else:
        raise ValueError(f"No model found with {ae_type}")
    ae.compute_output_shape(input_shape=(None, 100))
    return ae


class Autoencoder(tf.keras.Model, ABC):
    def __init__(self, shape, inter_dim, latent_dim, latent_slices, window_size, step_size, fc, batch_size, reduce, alpha, include_losses, opt):
        super(Autoencoder, self).__init__()
        self.nb_epochs = tf.Variable(0., trainable=False)
        self.fc = fc
        self.latent_dim = latent_dim
        self.slices = latent_slices
        self.shape = shape
        self.reduce = reduce
        self.alpha = alpha
        self.inter_dim = inter_dim
        self.window_size = window_size
        self.step_size = step_size

        self.encoder, self.decoder = self.create_encoder_decoder()
        self.losses_list = [tf.keras.metrics.Mean(name="total_loss"),
                            tf.keras.metrics.Mean(name="unbiased_loss"),
                            tf.keras.metrics.Mean(name="recon_loss"),
                            tf.keras.metrics.Mean(name="prior_loss"),
                            tf.keras.metrics.Mean(name="entropy")]
        self.batch_size = batch_size
        self.optimizer = opt
        self.include_losses = include_losses

    def get_config(self):
        config = super(Autoencoder, self).get_config()
        config.update({"fc": self.fc,
                       "latent_dim": self.latent_dim,
                       "latent_slices": self.slices,
                       "shape": self.shape,
                       "inter_dim": self.inter_dim,
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
                       "alpha": self.alpha,
                       "include_losses": self.include_losses,
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

    @ tf.function
    def call(self, x, training=False):
        processed = self.preprocess(x)
        encoded = self.encoder(processed, training=training)
        decoded = self.decoder(encoded, training=training)
        final_decoded = self.postprocess(decoded)
        return encoded, tf.keras.losses.mse(x, final_decoded)+tf.keras.losses.mae(x, final_decoded), final_decoded

    @ tf.function
    def custom_train_step(self, x):

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            encoded, recon_loss, _ = self.call(x, training=True)
            # Define a Keras Variable for \theta_ls

            theta = generate_theta(self.slices, self.latent_dim)

            # Define a Keras Variable for samples of z
            z = generate_z(encoded.shape[0], self.latent_dim, self.shape)
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

            recon_loss = self.reduce(recon_loss)

            drecon_dx = tape.gradient(recon_loss, x)

            pos_count = tf.cast(tf.math.count_nonzero(
                tf.math.greater_equal(drecon_dx, 0.), axis=0), dtype=tf.float32)

            pos_prob = pos_count/encoded.shape[0]

            # pos_prob = tf.cast(pos_prob, dtype=tf.float32)

            neg_prob = 1.0-pos_prob

            entropy = -(tf.math.multiply_no_nan(tf.math.log(pos_prob), pos_prob)
                        + tf.math.multiply_no_nan(tf.math.log(neg_prob), neg_prob))
            if tf.math.reduce_any(tf.math.is_nan(entropy)):
                tf.print(pos_prob, summarize=-1)
                tf.print(entropy, summarize=-1)

            entropy = tf.reduce_mean(entropy)

            loss = 0.
            if "recon_loss" in self.include_losses:
                loss += recon_loss
            if "entropy" in self.include_losses:
                loss += entropy
            if "sw_loss" in self.include_losses:
                loss += sw_loss*self.alpha

    # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

    # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.losses_list[0].update_state(loss)
        self.losses_list[1].update_state(sw_loss+recon_loss+entropy)
        self.losses_list[2].update_state(recon_loss)
        self.losses_list[3].update_state(sw_loss)
        self.losses_list[4].update_state(entropy)

        loss_results = {}
        for loss in self.losses_list:
            loss_results[loss.name] = loss.result()
        return loss_results

    @property
    def metrics(self):
        return self.losses_list

    @tf.function
    def increment_epoch(self):
        self.nb_epochs.assign_add(1.)
        self.reset_metrics()
        # return self.nb_epochs
        return tf.cast(self.nb_epochs, dtype=tf.int64)

    def reset_metrics(self):
        for loss in self.losses_list:
            loss.reset_state()


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

    @tf.function
    def preprocess(self, x):
        x = cluster_feature(x, self.fc)
        return generate_sliding_window(x, self.window_size, self.step_size)

    def postprocess(self, x):
        x = unslide_window(x, self.window_size,
                           self.step_size, self.batch_size)
        return uncluster_feature(x, self.fc)


class FC_AE(Autoencoder):
    def create_encoder_decoder(self):
        encoder = tf.keras.Sequential([
            layers.Dense(50, input_shape=(100,)),
            # layers.BatchNormalization(),
            layers.LeakyReLU(
                alpha=0.2, activity_regularizer=tf.keras.regularizers.L1(0.1)),
            layers.Dense(25),
            # layers.BatchNormalization(),
            layers.LeakyReLU(
                alpha=0.2, activity_regularizer=tf.keras.regularizers.L1(0.1)),
            layers.Dense(self.inter_dim),
            # layers.BatchNormalization(),
            layers.LeakyReLU(
                alpha=0.2, activity_regularizer=tf.keras.regularizers.L1(0.1)),
            layers.Dense(self.latent_dim)
        ])
        decoder = tf.keras.Sequential([
            layers.Dense(self.inter_dim, input_shape=(2,)),
            # layers.BatchNormalization(),
            layers.LeakyReLU(
                alpha=0.2, activity_regularizer=tf.keras.regularizers.L1(0.1)),
            layers.Dense(25),
            # layers.BatchNormalization(),
            layers.LeakyReLU(
                alpha=0.2, activity_regularizer=tf.keras.regularizers.L1(0.1)),
            layers.Dense(50),
            # layers.BatchNormalization(),
            layers.LeakyReLU(
                alpha=0.2, activity_regularizer=tf.keras.regularizers.L1(0.1)),
            layers.Dense(100, activation="relu")
        ])

        return encoder, decoder

    @tf.function
    def preprocess(self, x):
        x = cluster_feature(x, self.fc)
        return x

    @tf.function
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

    @tf.function
    def preprocess(self, x):
        x = cluster_feature(x, self.fc)
        return generate_sliding_window(x, self.window_size, self.step_size)

    @tf.function
    def postprocess(self, x):
        x = unslide_window(x, self.window_size,
                           self.step_size, self.batch_size)
        return uncluster_feature(x, self.fc)
