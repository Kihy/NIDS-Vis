from tqdm import tqdm
import pandas as pd
import pickle
from autoencoder import *
from helper import *
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import os
import io
from plotly.subplots import make_subplots
from itertools import product
from keract import get_activations
import plotly.express as px
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('WARNING')
# plt.style.use(['science', "no-latex"])
rng = default_rng()


def get_distance_map(autoencoder, resolution=(10, 10), top_right=(11, 1), bottom_left=(-1, -1)):

    # set up grid: xv,yv are latent coordiante locations, xi,yi are index of grid
    grid_points_x = tf.linspace(
        bottom_left[0], top_right[0], resolution[0])
    grid_points_y = tf.linspace(
        bottom_left[1], top_right[1], resolution[1])
    xv, yv = tf.meshgrid(grid_points_x, grid_points_y, indexing='ij')
    xi, yi = np.indices((resolution[0], resolution[1]), dtype=np.int64)

    # calculate corresponding decoded value
    # coords of shape resolution * resolution, 2
    coords = tf.reshape(
        tf.concat((xv[:, :, tf.newaxis], yv[:, :, tf.newaxis]), axis=-1), [-1, 2])

    #process in batches
    batch_size = 1024
    num_coords = coords.shape[0]
    split_sizes = [batch_size] * \
        (num_coords//batch_size)+[num_coords % batch_size]
    f_val = []
    for batch in tf.split(coords, split_sizes):

        f_val.append(autoencoder.postprocess(autoencoder.decoder(batch)))

    f_val = tf.concat(f_val, 0)
    f_val = tf.reshape(f_val, [resolution[0], resolution[1], -1])

    # set up filter for taking average
    filter_i = [-1, 0, 1]
    fx, fy = tf.meshgrid(filter_i, filter_i, indexing='ij')

    # calculate the index of all neighbours of each grid, boundary is extended.
    xi = xi[:, :, tf.newaxis, tf.newaxis]
    xi_n = tf.clip_by_value(xi+fx, 0, resolution[0]-1)
    yi = yi[:, :, tf.newaxis, tf.newaxis]
    yi_n = tf.clip_by_value(yi+fy, 0, resolution[1]-1)

    # concatenate neighbour coords
    coords_ni = tf.concat(
        (xi_n[:, :, :, :, tf.newaxis], yi_n[:, :, :, :, tf.newaxis]), axis=-1)
    # replace latent coord with decoded value
    neighbour_val = tf.gather_nd(f_val, coords_ni)

    # expand decode value for subtraction
    f_val = f_val[:, :, tf.newaxis, tf.newaxis, :]

    # calculate distance between neighbours in high dimensional space
    diff = neighbour_val-f_val

    mag_diff = tf.norm(diff, axis=-1)

    # calculate distance between neighbours in latent space
    dx = (top_right[0]-bottom_left[0])/(resolution[0]-1)
    dy = (top_right[1]-bottom_left[1])/(resolution[1]-1)
    dd = tf.sqrt(dx**2+dy**2)

    distance_matrix = tf.convert_to_tensor(
        [[dd, dy, dd], [dx, 1., dx], [dd, dy, dd]])

    mag_diff = mag_diff / \
        distance_matrix[tf.newaxis, tf.newaxis, :, :]

    max_idx_flat = tf.argmax(tf.reshape(
        mag_diff, [resolution[0], resolution[1], -1]), axis=-1)

    max_idx = tf.transpose(tf.unravel_index(
        tf.reshape(max_idx_flat, [-1]), [3, 3]))

    max_idx = tf.reshape(max_idx, [resolution[0], resolution[1], -1])
    diff_vec = tf.gather_nd(diff, max_idx, batch_dims=2)

    max_feature_idx = tf.argmax(diff_vec, axis=-1)

    max_feature_change = tf.gather_nd(
        diff_vec, max_feature_idx[:, :, tf.newaxis], batch_dims=2)

    # calculate average distance
    dist = tf.reduce_sum(mag_diff, axis=[-1, -2])/8.

    return grid_points_x, grid_points_y, dist, xv, yv, max_idx_flat, max_feature_idx, max_feature_change
    # return tf.reshape(xv, [-1]), tf.reshape(yv, [-1]), tf.reshape(dist/diff_lat, [-1])


def train_feature_cluster(path, chunksize):
    fc = corClust(100)

    ds = pd.read_csv(path, chunksize=chunksize,
                     usecols=list(range(100)),
                     header=None, skiprows=1, dtype=np.float32)
    # train correlation cluster
    for data in tqdm(ds):
        fc.update(data)

    return fc.cluster(100)[0]


def get_latent_position(autoencoder, scaler, path, frac=1):
    batch_size = 1024

    traffic_ds = get_dataset(path, batch_size, True,
                             scaler, frac, read_with="pd")
    # x = []
    # y = []
    latent_dim = []
    recon_array = []

    # Test autoencoder
    for data in tqdm(traffic_ds, leave=False, desc=f"Visualize: {path}"):

        latent, recon_error, decoded = autoencoder(data)
        #calculate reconstruction loss
        recon_array.extend(recon_error.numpy())
        latent_dim.extend(np.squeeze(latent))

    return np.array(latent_dim), recon_array


def linearity(latent, metric, verbose=False, name=""):
    latent = tf.abs(latent)

    min_lat = tf.reduce_min(latent, axis=0)
    max_lat = tf.reduce_max(latent, axis=0)

    latent_norm_x = (latent[:, 0]-min_lat[0])/(max_lat[0]-min_lat[0])
    latent_norm_y = (latent[:, 1]-min_lat[1])/(max_lat[1]-min_lat[1])

    latent = tf.stack([latent_norm_x, latent_norm_y], axis=1)

    top_right = tfp.stats.percentile(latent, 99.99, axis=0)
    bottom_left = tfp.stats.percentile(latent, 0.01, axis=0)

    # one diagonal
    diff = top_right-bottom_left
    m = diff[1]/diff[0]

    line_y = m*(latent[:, 0]-bottom_left[0])+bottom_left[1]
    line_y = line_y[:, tf.newaxis]
    latent_y = latent[:, 1][:, tf.newaxis]

    errors = metric(line_y, latent_y)
    mean_error = tf.reduce_mean(errors)

    if verbose:
        print(mean_error)

        plt.plot(latent[:, 0], line_y, label="line_y")
        plt.scatter(latent[:, 0], latent[:, 1],
                    label="latent", c=errors, alpha=0.04)

        # plt.legend()
        plt.colorbar()
        plt.savefig(f"exp_figs/{name}.png")
        plt.close()

    return mean_error


def get_mean_activation(autoencoder, x):
    activations = get_activations(
        autoencoder.encoder, x)
    # measure average model activation values
    mean_activations = tf.keras.metrics.Mean()
    for name, value in activations.items():
        if "leaky_re_lu" in name:
            mean_activations.update_state(tf.abs(value))
    return mean_activations.result()


def get_mean_weights(autoencoder):
    mean_weights = tf.keras.metrics.Mean()

    for layer in autoencoder.encoder.layers:
        weights = layer.get_weights()
        if not layer.name.startswith("dense"):
            continue
        w, b = weights
        mean_weights.update_state(tf.abs(w))
    return mean_weights.result()


def test_ae(batch_size, scaler, model_name, training_param, autoencoder=None, step=0):
    test_log_dir = f'logs/{model_name}/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    if autoencoder is None:
        autoencoder = tf.keras.models.load_model(
            f"../models/{model_name}")

    threshold = 0.

    #calculate mean activation
    mean_activations = tf.keras.metrics.Mean()

    for path in training_param["train_paths"]:
        # traffic_ds = pd.read_csv(path, chunksize=batch_size,
        #                          usecols=list(range(100)), header=None, skiprows=1, dtype=np.float32)
        traffic_ds = get_dataset(
            path, batch_size, shuffle=False, scaler=scaler)
        for x in tqdm(traffic_ds, leave=False, desc="Test Benign"):

            latent, err, _ = autoencoder(x)
            threshold = tf.maximum(threshold, tf.reduce_max(err))
            mean_activations.update_state(get_mean_activation(autoencoder, x))

    with test_summary_writer.as_default():
        tf.summary.scalar(
            f'activations/benign', mean_activations.result(), step=step)
        tf.summary.scalar(
            f'weights', get_mean_weights(autoencoder), step=step)

    #calculate linearity
    msle = tf.keras.metrics.Mean()
    linearity_metric = tf.keras.losses.MeanAbsoluteError(
        reduction=tf.keras.losses.Reduction.NONE)

    # calculate MDR
    mdr = tf.keras.metrics.Mean()
    for name, path in training_param["test_paths"].items():

        traffic_ds = get_dataset(
            path, batch_size, shuffle=False, scaler=scaler)

        latent_array = []
        for x in tqdm(traffic_ds, leave=False, desc=f"Test {name}"):

            latent, err, _ = autoencoder(x)
            mdr.update_state(
                tf.math.greater(err, threshold))
            latent_array.append(latent)
        latent_array = tf.concat(latent_array, axis=0)

        lin_m = linearity(latent_array, linearity_metric,
                          name=f"{model_name}_{name}")
        msle.update_state(lin_m)

        with test_summary_writer.as_default():
            tf.summary.scalar(
                f'MDR/{name}', mdr.result(), step=step)

            tf.summary.scalar(
                f'linearity/{name}', msle.result(), step=step)

        msle.reset_state()
        mdr.reset_state()


def train(ae_type, shape, ae_param, training_param, scaler, model_name):
    n_out = (ae_param["batch_size"]
             - ae_param["window_size"])/ae_param["step_size"]
    if int(n_out) != n_out:
        raise ValueError(
            "The step size and window size cannot nicely split the data."
            + "Ensure that (batch size-window size) / step size is an integer"+f"batch_size: {batch_size}, window_size {window_size}, step_size {step_size}")

    if training_param["continue_training"]:
        autoencoder = tf.keras.models.load_model(
            f"../models/{model_name}",
            custom_objects={"Autoencoder": Autoencoder}, compile=False)
    else:
        autoencoder = get_ae(ae_type, shape, ae_param)

    train_ae(autoencoder, ae_param, training_param, scaler, model_name)

    tf.keras.models.save_model(
        autoencoder, f"../models/{model_name}", options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))


def train_ae(autoencoder, ae_param, training_param, scaler, model_name):

    train_log_dir = f'logs/{model_name}/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    traffic_ds = get_dataset(
        training_param["train_paths"], ae_param["batch_size"], training_param["shuffle"], scaler)

    for i in tqdm(range(training_param["epochs"])):

        for x in tqdm(traffic_ds, leave=False, position=1, desc="Train"):
            hist = autoencoder.custom_train_step(x)

        nb_epochs = autoencoder.increment_epoch()

        with train_summary_writer.as_default():
            for key, value in hist.items():
                tf.summary.scalar(f"losses/{key}", value, step=nb_epochs)

        if nb_epochs == 1 or nb_epochs == training_param["epochs"] or nb_epochs % training_param["test_iter"] == 0:
            test_ae(2**10,
                    scaler, model_name, training_param, autoencoder=autoencoder, step=nb_epochs)


def visualise_latent_space(scaler, eval_param, model_name, feature_names, plot_dist=False, step=0):
    autoencoder = tf.keras.models.load_model(
        f"../models/{model_name}", compile=False)

    fig = make_subplots(rows=1, cols=2)

    scales = []
    distance = 1.2
    for name, info in tqdm(eval_param["feature_paths"].items()):

        if name == "Prior":

            fig.add_shape(type="circle",
                          xref="x", yref="y",
                          x0=-1, y0=-1,
                          x1=1, y1=1,
                          opacity=0.2,
                          fillcolor="orange",
                          line_color="orange",
                          )
        else:
            if info[1] not in scales:
                show_scale = True
                distance += 0.1
                # scales.append(info[1])
            else:
                show_scale = False
            if info[0].endswith("scaled.csv"):
                latent, recon = get_latent_position(
                    autoencoder, None, info[0], eval_param["frac"])
            else:
                latent, recon = get_latent_position(
                    autoencoder, scaler, info[0], eval_param["frac"])
            average_recon = np.mean(recon)

            if name == "Benign":

                benign_recon = average_recon
                benign_latent = latent
                benign_recons = recon

            fig.add_trace(go.Scattergl(x=latent[:, 0], y=latent[:, 1],
                                       mode='markers',
                                       opacity=0.5,
                                       name=f"{name}_{average_recon:.3f}",
                                       text=[f"{i}_{recon[i]:.3f}" for i in range(
                                           latent.shape[0])],
                                       legendgroup=f"{name}_{average_recon:.3f}",
                                       marker=dict(
                size=10,
                # set color equal to a variable
                color=recon,
                colorscale=info[1],  # one of plotly colorscales
                showscale=show_scale,
                colorbar=dict(x=distance))
            ), row=1, col=1)

            fig.add_trace(go.Scattergl(x=latent[:, 0], y=latent[:, 1],
                                       mode='markers',
                                       opacity=0.5,
                                       name=f"{name}_{average_recon:.3f}",
                                       legendgroup=f"{name}_{average_recon:.3f}",
                                       text=[f"{i}_{recon[i]:.3f}" for i in range(
                                           latent.shape[0])],
                                       marker=dict(
                size=10,
                # set color equal to a variable
                color=list(range(latent.shape[0])),
                colorscale=info[1],  # one of plotly colorscales
                showscale=False)
            ), row=1, col=2)

    if plot_dist:
        av_dist = plot_latent_similarity(eval_param["resolution"], autoencoder,
                                         benign_latent, benign_recons, model_name, feature_names)
        test_log_dir = f'logs/{model_name}/test'
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        with test_summary_writer.as_default():
            tf.summary.scalar(
                'similarity', av_dist, step=step)

    fig.write_html(f"../plots/{model_name}.html")
    return benign_recon


def plot_latent_similarity(resolution, autoencoder, benign_latent, benign_recons, model_name, feature_names, dir_name="../plots", return_scatter=False):
    fig = go.Figure()

    top_right = tf.reduce_max(benign_latent, axis=0)

    bottom_left = tf.reduce_min(benign_latent, axis=0)

    x_grid, y_grid, dist, x, y, max_idx, max_feature_idx, max_feature_change = get_distance_map(
        autoencoder, resolution=resolution, top_right=top_right, bottom_left=bottom_left)

    average_distance = tfp.stats.percentile(
        dist, 50., interpolation="midpoint")
    fig.update_layout(title=f"Median Distance: {average_distance:.4f}")

    contour = go.Contour(x=x_grid, y=y_grid,
                         z=tf.transpose(dist),
                         name="distance map",
                         text=tf.transpose(dist),
                         line_smoothing=0.85,
                         contours_showlines=False,
                         colorbar=dict(x=1.2),
                         colorscale="Brwnyl",
                         opacity=0.5,
                         )

    latent = go.Scattergl(x=benign_latent[:, 0], y=benign_latent[:, 1],
                          mode='markers',
                          opacity=0.5,
                          name="benign",
                          text=[f"{i}_{benign_recons[i]:.3f}" for i in range(
                                   benign_latent.shape[0])],
                          marker=dict(
        size=10,
        # set color equal to a variable
        color=benign_recons,
        colorscale="Agsunset",  # one of plotly colorscales
        showscale=True,
        colorbar=dict(x=1.3))
    )

    max_feature_idx = tf.gather(
        feature_names, tf.reshape(max_feature_idx, [-1]))

    max_idx = tf.gather([12, 5, 9, 7, 0, 8, 11, 6, 10],
                        tf.reshape(max_idx, [-1]))

    scatter = px.scatter(x=tf.reshape(x, [-1]), y=tf.reshape(y, [-1]),
                         opacity=0.5,
                         # name="max feature",
                         hover_name=[f"{max_feature_idx[i]}" for i in range(len(
                                   max_feature_idx))],
                         color=max_feature_idx,
                         symbol=max_idx,
                         symbol_map='identity',

                         size=tf.reshape(max_feature_change, [-1])*100
                         )

    if return_scatter:
        return contour, latent, scatter.data

    fig.add_trace(contour)
    fig.add_trace(latent)
    fig.add_traces(scatter.data)

    fig.write_html(f"{dir_name}/{model_name}_dist.html")
    print("saved at:", f"{dir_name}/{model_name}_dist.html")
    return average_distance


if __name__ == '__main__':

    alphas = [10.0]
    losses = [
        # ["recon_loss"],
        # ["recon_loss", "sw_loss"],
        ["recon_loss", "entropy", "sw_loss"],
        # ["recon_loss", "distance_loss", "sw_loss"]
              ]
    reduce_types = {"mean": tf.reduce_mean}
    optimizers = {
        "amsgrad": tf.keras.optimizers.Adam(amsgrad=True),
        "adam": tf.keras.optimizers.Adam(),
        "rmsprop": tf.keras.optimizers.RMSprop(),
        # "sgd": tf.keras.optimizers.SGD()
                  }

    scaler_path = "../../mtd_defence/models/uq/autoencoder/Cam_1_scaler.pkl"
    training_param = {"train_paths": ["../../mtd_defence/datasets/uq/benign/Cam_1.csv"],
                      "test_paths": {
                        "ACK": "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding.csv",
                        "SYN": "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_SYN_Flooding.csv",
                        "UDP": "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_UDP_Flooding.csv"},
                      "continue_training": False,
                      "test_iter": 30,
                      "shuffle": True,
                      "epochs": 1000}

    if not os.path.exists("../plots"):
        os.mkdir("../plots")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    if not training_param["continue_training"]:
        fc = train_feature_cluster(training_param["train_paths"][0], 2**10)
    else:
        fc = None

    count = 0

    feature_names = ["HT_MI_5_weight", "HT_MI_5_mean", "HT_MI_5_std", "HT_MI_3_weight", "HT_MI_3_mean", "HT_MI_3_std",
                     "HT_MI_1_weight", "HT_MI_1_mean", "HT_MI_1_std", "HT_MI_0.1_weight",
                     "HT_MI_0.1_mean", "HT_MI_0.1_std", "HT_MI_0.01_weight", "HT_MI_0.01_mean",
                     "HT_MI_0.01_std", "HT_H_5_weight", "HT_H_5_mean", "HT_H_5_std", "HT_H_5_radius",
                     "HT_H_5_magnitude", "HT_H_5_covariance", "HT_H_5_pcc", "HT_H_3_weight", "HT_H_3_mean",
                     "HT_H_3_std", "HT_H_3_radius", "HT_H_3_magnitude", "HT_H_3_covariance", "HT_H_3_pcc",
                     "HT_H_1_weight", "HT_H_1_mean", "HT_H_1_std", "HT_H_1_radius", "HT_H_1_magnitude", "HT_H_1_covariance",
                     "HT_H_1_pcc", "HT_H_0.1_weight", "HT_H_0.1_mean", "HT_H_0.1_std", "HT_H_0.1_radius", "HT_H_0.1_magnitude",
                     "HT_H_0.1_covariance", "HT_H_0.1_pcc", "HT_H_0.01_weight", "HT_H_0.01_mean", "HT_H_0.01_std",
                     "HT_H_0.01_radius", "HT_H_0.01_magnitude", "HT_H_0.01_covariance", "HT_H_0.01_pcc", "HT_jit_5_weight",
                     "HT_jit_5_mean", "HT_jit_5_std", "HT_jit_3_weight", "HT_jit_3_mean", "HT_jit_3_std", "HT_jit_1_weight",
                     "HT_jit_1_mean", "HT_jit_1_std", "HT_jit_0.1_weight", "HT_jit_0.1_mean", "HT_jit_0.1_std", "HT_jit_0.01_weight",
                     "HT_jit_0.01_mean", "HT_jit_0.01_std", "HT_Hp_5_weight", "HT_Hp_5_mean", "HT_Hp_5_std", "HT_Hp_5_radius",
                     "HT_Hp_5_magnitude", "HT_Hp_5_covariance", "HT_Hp_5_pcc", "HT_Hp_3_weight", "HT_Hp_3_mean", "HT_Hp_3_std",
                     "HT_Hp_3_radius", "HT_Hp_3_magnitude", "HT_Hp_3_covariance", "HT_Hp_3_pcc", "HT_Hp_1_weight", "HT_Hp_1_mean",
                     "HT_Hp_1_std", "HT_Hp_1_radius", "HT_Hp_1_magnitude", "HT_Hp_1_covariance", "HT_Hp_1_pcc", "HT_Hp_0.1_weight",
                     "HT_Hp_0.1_mean", "HT_Hp_0.1_std", "HT_Hp_0.1_radius", "HT_Hp_0.1_magnitude", "HT_Hp_0.1_covariance",
                     "HT_Hp_0.1_pcc", "HT_Hp_0.01_weight", "HT_Hp_0.01_mean", "HT_Hp_0.01_std", "HT_Hp_0.01_radius",
                     "HT_Hp_0.01_magnitude", "HT_Hp_0.01_covariance", "HT_Hp_0.01_pcc"]

    feature_names = [i[:i.rindex("_")] for i in feature_names]

    eval_param = {"feature_paths": {
                  "Benign": ["../../mtd_defence/datasets/uq/benign/Cam_1.csv", "deep"],
                  "ACK": ["../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding.csv", "matter"],
                  "ACK_Scaled": ["../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding_scaled.csv", "matter"],
                  "SYN": ["../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_SYN_Flooding.csv", "matter"],
                  "UDP": ["../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_UDP_Flooding.csv", "matter"],
                  },
                  "frac": 0.05,
                  "resolution": (200, 200),
                  "batch_size": 2**10}

    # for model_name in os.listdir("../models"):
    #     test_ae(2**12, scaler, model_name,
    #             training_param, step=1000)
    #     print("finish", model_name)

    for alpha, loss, reduce_type, optimizer in product(alphas, losses, reduce_types.items(), optimizers.items()):
        reduce_name, reduce = reduce_type
        opt_name, opt = optimizer

        ae_param = {"inter_dim": 16,
                    "latent_dim": 2,
                    "latent_slices": 200,
                    "window_size": 64,
                    "step_size": 1,
                    "batch_size": 2**10,
                    "fc": fc,
                    "reduce": reduce,
                    "alpha": alpha,
                    "include_losses": loss,
                    "opt": opt,
                    }

        model_name = f"FC_circular_{reduce_name}_{alpha}_{'_'.join(loss)}_{opt_name}_act_1.0"
        print(f"training {model_name}")
        train("FC", "circular", ae_param,
              training_param, scaler, model_name)
        # print(opt)
        # test_ae(2**12, scaler, model_name,
        #         training_param, step=1000, verbose=True)

        visualise_latent_space(
                            scaler, eval_param, model_name, feature_names, plot_dist=True, step=training_param["epochs"])
