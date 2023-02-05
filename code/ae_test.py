import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from gudhi.tensorflow import RipsLayer
from sklearn.datasets import make_blobs
from itertools import chain, combinations
from sklearn.manifold import TSNE


plt.rcParams['axes.facecolor'] = 'white'


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


@tf.function
def generate_theta(n, dim, dtype="float32"):
    """generates n slices to be used to slice latent space, with dim dimensions. the slices are unit vectors"""
    theta, _ = tf.linalg.normalize(tf.random.normal(
        shape=[n, dim], dtype=dtype), axis=1)
    return theta


@tf.function
def generate_z(n, dim, shape, dtype="float32", center=0):
    """generates n samples with shape in dim dimensions, represents the prior distribution"""
    if shape == "uniform":
        z = 2 * tf.random.uniform(shape=[n, dim], dtype=dtype) - 1
    elif shape == "circular":
        u = tf.random.normal(shape=[n, dim], dtype=dtype)
        normalised_u, norm = tf.linalg.normalize(u, axis=1)
        r = tf.random.uniform(shape=[n], dtype=dtype)**(1.0 / dim)
        z = tf.expand_dims(r, axis=1) * normalised_u
        z += center
    return z


@tf.function
def pairwise_distance(x):
    x_flat = tf.reshape(x, [x.shape[0], -1])
    return tf.norm(x_flat[:, None] - x_flat, ord='euclidean', axis=-1)


@tf.function(reduce_retracing=True)
def sig_error(signature1, signature2):
    """Compute distance between two topological signatures."""
    d0_death_diff = tf.keras.losses.mean_squared_error(
        signature1 / tf.reduce_max(signature1), signature2 / tf.reduce_max(signature2))

    # d1_death_diff = tf.keras.losses.mean_squared_error(
    #     signature1[1]["death"] / s1_dim, signature2[1]["death"] / s2_dim)
    # d1_birth_diff = tf.keras.losses.mean_squared_error(
    #     signature1[1]["birth"] / s1_dim, signature2[1]["birth"] / s2_dim)

    return d0_death_diff  # + d1_death_diff + d1_birth_diff


def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)
    return decorator


class Autoencoder(tf.keras.Model):
    def __init__(self, losses=["recon"]):
        super().__init__()
        self.input_dim = 10
        self.latent_dim = 2
        self.slices = 128
        self.encoder, self.decoder = self.create_encoder_decoder([5])
        self.rl = RipsLayer(homology_dimensions=[0])
        self.use_losses = losses
        self.losses_list = {loss_name: tf.keras.metrics.Mean()
                            for loss_name in self.use_losses}
        self.optimizer = tf.keras.optimizers.get({"class_name": "Adam",
                                                 "config": {"learning_rate": 1e-3}})

    @tf.function
    def call(self, x):
        encode_x = self.encoder(x)
        decoded_x = self.decoder(encode_x)
        recon_loss = tf.keras.losses.mse(x, decoded_x)

        return encode_x, recon_loss, decoded_x

    # @tf.function
    def train_step(self, x):

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)

            encoded, recon_loss, decoded_x = self.call(x)

            total_loss = 0.
            for loss in self.use_losses:
                if loss == "recon":
                    loss_value = tf.reduce_mean(recon_loss)

                if loss == "dist_loss":

                    lat_distance = pairwise_distance(encoded)
                    x_distance = pairwise_distance(x)
                    loss_value = tf.reduce_mean(
                        tf.keras.losses.mse(lat_distance, x_distance))

                if loss == "ranking_loss":

                    lat_distance = tf.norm(encoded, axis=1)
                    normalised_lat = (lat_distance - tf.reduce_min(lat_distance)) / (
                        tf.reduce_max(lat_distance) - tf.reduce_min(lat_distance))
                    normalised_recon = (recon_loss - tf.reduce_min(recon_loss)) / (
                        tf.reduce_max(recon_loss) - tf.reduce_min(recon_loss))
                    distance1 = normalised_lat[None,
                                               :] - normalised_lat[:, None]
                    recon1 = normalised_recon[None,
                                              :] - normalised_recon[:, None]
                    loss_value = tf.reduce_sum(
                        tf.nn.relu(-distance1 * recon1), axis=1)

                if loss == "sw_loss":
                    theta = generate_theta(
                        self.slices, self.latent_dim, dtype=encoded.dtype)

                    # Define a Keras Variable for samples of z
                    z = generate_z(encoded.shape[0], self.latent_dim,
                                   "circular", dtype=encoded.dtype, center=0)

                    # Let projae be the projection of the encoded samples
                    projae = tf.tensordot(encoded, tf.transpose(theta), axes=1)
                    # projae += tf.expand_dims(tf.norm(encoded, axis=1), axis=1)
                    # Let projz be the projection of the $q_Z$ samples
                    projz = tf.tensordot(z, tf.transpose(theta), axes=1)
                    # projz += tf.expand_dims(tf.norm(z, axis=1), axis=1)
                    # Calculate the Sliced Wasserstein distance by sorting
                    # the projections and calculating the L2 distance between
                    loss_value = tf.reduce_mean((tf.sort(tf.transpose(projae))
                                                 - tf.sort(tf.transpose(projz)))**2)
                if loss == "topological_loss":
                    x_dist = pairwise_distance(x)
                    latent_dist = pairwise_distance(encoded)

                    x_sig = self.rl(x)[0][0]
                    latent_sig = self.rl(encoded)[0][0]
                    x_pairs = self.get_edge_idx(x_dist, x_sig[:, 1])
                    latent_pairs = self.get_edge_idx(
                        latent_dist, latent_sig[:, 1])

                    loss_value = sig_error(tf.gather_nd(latent_dist, x_pairs), tf.gather_nd(x_dist, x_pairs)) +\
                        sig_error(tf.gather_nd(latent_dist, latent_pairs),
                                  tf.gather_nd(x_dist, latent_pairs))
                total_loss += loss_value
                self.losses_list[loss].update_state(loss_value)
        # Compute gradients
        trainable_vars = self.trainable_variables

        gradients = tape.gradient(total_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

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


dataset_size = 51200


X, y_true, centers = make_blobs(
    n_samples=dataset_size, n_features=10, centers=4, cluster_std=4, random_state=12, return_centers=True)


# non-linearly transform some features
X[:, 0] = np.log(np.abs(X[:, 0])) * np.sign(X[:, 0])
centers[:, 0] = np.log(np.abs(centers[:, 0])) * np.sign(centers[:, 0])
X[:, 4] = np.exp(X[:, 4])
centers[:, 4] = np.exp(centers[:, 4])
X[:, 9] = np.power(X[:, 9], 2)
centers[:, 9] = np.power(centers[:, 9], 2)
X[:, 7] = np.sqrt(np.abs(X[:, 7])) * np.sign(X[:, 7])
centers[:, 7] = np.sqrt(np.abs(centers[:, 7])) * np.sign(centers[:, 7])


scaler = MinMaxScaler()
data = scaler.fit_transform(X)

train_data = pd.DataFrame(data=data)

train_data = tf.cast(train_data, tf.float32)

perplexities = [3, 5, 10, 30, 50, 100]

# for p in perplexities:
#     tsne = TSNE(n_components=2, learning_rate='auto',
#                 init='random', perplexity=p)
#
#     tnc = np.vstack([train_data, centers])
#     train_embedded = tsne.fit_transform(tnc)
#
#     fig, axs = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
#
#     sc2 = axs.scatter(train_embedded[:-4, 0], train_embedded[:-4, 1],
#                       c=y_true, alpha=0.3, label=y_true)
#     print(train_embedded[-5:])
#     axs.scatter(train_embedded[-5:, 0], train_embedded[-5:, 1], color="red")
#     axs.set_title("classes")
#     legend1 = axs.legend(*sc2.legend_elements(), loc='center left',
#                          bbox_to_anchor=(1., 0.5), title="Classes")
#     axs.add_artist(legend1)
#     plt.tight_layout()
#     plt.savefig(f"exp_figs/ae_loss/tsne_{p}_latent.png")
#
for losses in powerset(["topological_loss", "dist_loss", "recon", "sw_loss", "ranking_loss"]):
    if "topological_loss" not in losses:
        continue
    losses = list(losses)

    model1 = Autoencoder(losses)
    model1.compile()

    # autoencoder = AnomalyDetector()

    # autoencoder.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True))
    if "topological_loss" in losses:
        epochs = 200
    else:
        epochs = 500
    history = model1.fit(train_data,
                         epochs=epochs,
                         batch_size=128,
                         shuffle=True,
                         verbose=2)

    fig, axs = plt.subplots(len(history.history), 1, figsize=(
        5, len(history.history) * 3), facecolor='white')

    if len(losses) == 1:
        name = losses[0]
        axs.plot(history.history[name], label=name)
        axs.set_yscale('log')
        axs.set_title(f"{name}: {history.history[name][-1]:.4f}")
    else:
        i = 0
        for name, value in history.history.items():
            axs[i].plot(value, label=name)
            axs[i].set_yscale('log')
            axs[i].set_title(f"{name}: {history.history[name][-1]:.4f}")
            i += 1
    plt.tight_layout()
    plt.savefig(f"exp_figs/ae_loss/{'_'.join(losses)}_losses.png")

    scaled_centers = scaler.transform(centers)
    latent_centers, _, _ = model1(scaled_centers)
    print(latent_centers)

    latent, recon, _ = model1(train_data)
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), facecolor='white')

    sc = axs[0].scatter(latent[:, 0], latent[:, 1], c=recon, alpha=0.3)
    axs[0].set_title("reconstruction error")
    fig.colorbar(sc, ax=axs[0])
    sc2 = axs[1].scatter(latent[:, 0], latent[:, 1],
                         c=y_true, alpha=0.3, label=y_true)
    axs[1].scatter(latent_centers[:, 0], latent_centers[:, 1], color="red")
    axs[1].set_title("classes")
    legend1 = axs[1].legend(*sc2.legend_elements(), loc='center left',
                            bbox_to_anchor=(1., 0.5), title="Classes")
    axs[1].add_artist(legend1)
    plt.tight_layout()
    plt.savefig(f"exp_figs/ae_loss/{'_'.join(losses)}_latent.png")
    # del model1
    tf.keras.backend.clear_session()
