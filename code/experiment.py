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
import plotly.express as px
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding,Isomap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('WARNING')
# plt.style.use(['science', "no-latex"])
rng = default_rng()

device_name = tf.test.gpu_device_name()
if not device_name:
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


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
        (num_coords // batch_size) + [num_coords % batch_size]
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
    xi_n = tf.clip_by_value(xi + fx, 0, resolution[0] - 1)
    yi = yi[:, :, tf.newaxis, tf.newaxis]
    yi_n = tf.clip_by_value(yi + fy, 0, resolution[1] - 1)

    # concatenate neighbour coords
    coords_ni = tf.concat(
        (xi_n[:, :, :, :, tf.newaxis], yi_n[:, :, :, :, tf.newaxis]), axis=-1)
    # replace latent coord with decoded value
    neighbour_val = tf.gather_nd(f_val, coords_ni)

    # expand decode value for subtraction
    f_val = f_val[:, :, tf.newaxis, tf.newaxis, :]

    # calculate distance between neighbours in high dimensional space
    diff = neighbour_val - f_val

    mag_diff = tf.norm(diff, axis=-1)

    # calculate distance between neighbours in latent space
    dx = (top_right[0] - bottom_left[0]) / (resolution[0] - 1)
    dy = (top_right[1] - bottom_left[1]) / (resolution[1] - 1)
    dd = tf.sqrt(dx**2 + dy**2)

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
    dist = tf.reduce_sum(mag_diff, axis=[-1, -2]) / 8.

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


def linearity(latent, metric, verbose=False, name=""):
    latent = tf.abs(latent)

    min_lat = tf.reduce_min(latent, axis=0)
    max_lat = tf.reduce_max(latent, axis=0)

    latent_norm_x = (latent[:, 0] - min_lat[0]) / (max_lat[0] - min_lat[0])
    latent_norm_y = (latent[:, 1] - min_lat[1]) / (max_lat[1] - min_lat[1])

    latent = tf.stack([latent_norm_x, latent_norm_y], axis=1)

    top_right = tfp.stats.percentile(latent, 99.99, axis=0)
    bottom_left = tfp.stats.percentile(latent, 0.01, axis=0)

    # one diagonal
    diff = top_right - bottom_left
    m = diff[1] / diff[0]

    line_y = m * (latent[:, 0] - bottom_left[0]) + bottom_left[1]
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
        if not layer.name.startswith("dense"):
            continue
        weights = layer.get_weights()
        if layer.get_config()["use_bias"]:
            w, b = weights
        else:
            w = weights
        mean_weights.update_state(tf.abs(w))

    for layer in autoencoder.decoder.layers:
        if not layer.name.startswith("dense"):
            continue
        weights = layer.get_weights()

        if layer.get_config()["use_bias"]:
            w, b = weights
        else:
            w = weights
        mean_weights.update_state(tf.abs(w))
    return mean_weights.result()


def test_ae(batch_size, scaler, model_name, training_param, autoencoder=None, step=0):
    test_log_dir = f'logs/{model_name}/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    if autoencoder is None:
        autoencoder = tf.keras.models.load_model(
            f"../models/{model_name}", custom_objects={"LogMinMaxScaler": LogMinMaxScaler})

    threshold = 0.

    # calculate mean activation
    mean_activations = tf.keras.metrics.Mean()
    mean_mae = tf.keras.metrics.Mean()
    mae = tf.keras.losses.MeanAbsoluteError()

    for path in training_param["train_paths"]:
        # traffic_ds = pd.read_csv(path, chunksize=batch_size,
        #                          usecols=list(range(100)), header=None, skiprows=1, dtype=np.float32)
        traffic_ds = get_dataset(
            path, batch_size, shuffle=False, scaler=scaler)
        for x in tqdm(traffic_ds, leave=False, desc="Test Benign"):

            latent, err, decoded_x = autoencoder(x)
            threshold = tf.maximum(threshold, tf.reduce_max(err))
            mean_activations.update_state(get_mean_activation(autoencoder, x))
            mean_mae.update_state(mae(x, decoded_x))

    with test_summary_writer.as_default():
        tf.summary.scalar(
            f'activations/benign', mean_activations.result(), step=step)
        tf.summary.scalar(
            f'weights', get_mean_weights(autoencoder), step=step)
        tf.summary.scalar(
            f'mae/benign', mean_mae.result(), step=step)

    # calculate linearity
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


def train(dr_type, dr_param, training_param):
    if dr_type == "pkl":
        if dr_param["model_name"]=="pca":
            dr_model = PCA(**dr_param["config"])
        elif dr_param["model_name"]=="kernel_pca":
            dr_model = KernelPCA(**dr_param["config"])
        elif dr_param["model_name"]=="lle":
            dr_model=LocallyLinearEmbedding(**dr_param["config"])
        elif dr_param["model_name"]=="isomap":
            dr_model=Isomap(**dr_param["config"])
        elif dr_param["model_name"]=="umap":
            dr_model = umap.UMAP(**dr_param["config"])
        elif dr_param["model_name"]=="tsne":
            dr_model = TSNE(**dr_param["config"])
        else:
            raise ValueError("Invalid dimensionality reduction type")
        dr_model=train_sklearn_model(dr_model, training_param)
        with open(f"../models/{dr_param['model_name']}.pkl", "wb") as of:
            pickle.dump(dr_model, of)

    else:

        if training_param["continue_training"]:
            autoencoder = tf.keras.models.load_model(
                f"../models/{dr_param['model_name']}",
                custom_objects={"Autoencoder": Autoencoder},  compile=False)
        else:
            autoencoder = get_ae(dr_type, dr_param)

        train_tf_model(autoencoder,
                       training_param, None, dr_param['model_name'])

        


def train_sklearn_model(dr_model, training_param):
    traffic_ds = get_dataset(**training_param)

    if training_param["batch_size"] > 0:
        for x in tqdm(traffic_ds, leave=False, position=1, desc="Train"):
            dr_model.partial_fit(x)
    else:
        dr_model=dr_model.fit(traffic_ds)
    return dr_model

def train_tf_model(dr_model, training_param, scaler, model_name, dtype="float32"):

    train_log_dir = f'logs/{model_name}/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    traffic_ds = get_dataset(
        training_param["train_paths"], training_param["batch_size"], dr_model.input_dim, training_param["shuffle"], scaler,  dtype=dtype)

    
    for i in tqdm(range(training_param["epochs"])):
        for x in tqdm(traffic_ds, leave=False, position=1, desc="Train"):
            
            hist = dr_model.custom_train_step(x)

        nb_epochs = dr_model.increment_epoch()

        with train_summary_writer.as_default():
            for key, value in hist.items():
                tf.summary.scalar(f"losses/{key}", value, step=nb_epochs)

        # if nb_epochs == 1 or nb_epochs == training_param["epochs"] or nb_epochs % training_param["test_iter"] == 0:
        #     test_ae(2**10,
        #             scaler, model_name, training_param, autoencoder=dr_model, step=nb_epochs)
        if i+1 in training_param["save_epochs"]:
            tf.keras.models.save_model(
            dr_model, f"../models/{training_param['dataset']}/{model_name}_{nb_epochs}", options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))
            print(f"../models/{training_param['dataset']}/{model_name}_{nb_epochs}")

def visualise_latent_space(model_type, model_name, scaler, feature_paths, feature_names, resolution=(100,100),
                           plot_dist=False, step=0, frac=1, dtype="float32"):
    
    if model_type=="pkl":
        dr_model=GenericDRModel(**{"name":model_name, "save_type":model_type, "path":f"../models/{model_name}.pkl",
                        "func_name":"transform",  "scaler":scaler,"threshold":0.3})
    else:
        dr_model=HybridModel(**{"name":model_name, "save_type":model_type, "path":f"../models/{model_name}",
                        "func_name": "call", "ad_output_index":1, "dr_output_index":0,"scaler":scaler,"threshold":0.3,"dtype":dtype})

    fig = make_subplots(rows=1, cols=2)

    scales = []
    distance = 1.2
    for name, info in tqdm(feature_paths.items()):

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

            
            latent, recon= get_latent_position(
                dr_model, None, info[0], frac=frac, dtype=dtype)
            
           
            average_recon = np.mean(recon)
            if name == "Benign":
                benign_recon = average_recon
                benign_latent = latent
                benign_recons = recon
            print(recon.shape)
            if len(recon)>0:
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
                                       name=f"{name}",
                                       legendgroup=f"{name}",
                                       marker=dict(
                size=10,
                # set color equal to a variable
                color=list(range(latent.shape[0])),
                colorscale=info[1],  # one of plotly colorscales
                showscale=False)
            ), row=1, col=2)

    if plot_dist:
        av_dist = plot_latent_similarity(resolution, dr_model,
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

                         size=tf.reshape(max_feature_change, [-1]) * 100
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
    
    devices=["Cam_1", "Google-Nest-Mini_1","Lenovo_Bulb_1", "Raspberry_Pi_telnet", "Smart_Clock_1","Smartphone_1","SmartTV"]

    filtered=False # True, 
    denoise = [False] #,True
    activation=["sigmoid"] #,"relu"

    lat_dims=[25] #2,
    alpha = 10.
    loss=["recon_dist_loss"] #,
    reduce_types = {"mean": tf.reduce_mean}
    lr = 1e-3
    optimizers = {
        "adam": {"class_name": "Adam",
                 "config": {"learning_rate": lr}}
        }

    for dataset in devices:
        scaler_type = "min_max"
        tf_scaler_path=f"../../mtd_defence/models/uq/autoencoder/{dataset}_{scaler_type}_scaler.pkl"
        with open(tf_scaler_path, "rb") as f:
            tf_scaler = pickle.load(f)
        
        sk_scaler_path = f"../../mtd_defence/models/uq/autoencoder/{dataset}_scaler.pkl"
        with open(sk_scaler_path, "rb") as f:
            sk_scaler = pickle.load(f)
        
        if not os.path.exists("../plots"):
            os.mkdir("../plots")

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

        # model_name = "tsne"
        # save_type="pkl"
        # sk_train_param = {"path": "../../mtd_defence/datasets/uq/benign/Cam_1.csv",
        #                   "frac": 0.1, "read_with": "pd", "scaler": sk_scaler,
        #                   "batch_size": -1,
        #                   }
        # eval_param = {"model_type": save_type,
        #               "model_name":model_name,
        #               "feature_paths": {
        #                   "Benign": ["../../mtd_defence/datasets/uq/benign/Cam_1.csv", "deep"],
        #                   "ACK": ["../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding.csv", "matter"],
        #                 #   "ACK_adv": ["../../mtd_defence/datasets/uq/adversarial/Cam_1/ACK_Flooding/autoencoder_0.5_10_3_False_pso0.5/csv/Cam_1_ACK_Flooding_iter_0.csv", "matter"],
        #                 #   "SYN": ["../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_SYN_Flooding.csv", "matter"],
        #                 #   "UDP": ["../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_UDP_Flooding.csv", "matter"],
        #               },
        #               "scaler":None,
        #               "frac": 0.05,
        #               "feature_names":feature_names,
        #               "plot_dist":False,
        #               "resolution": (200, 200),
        #              }
        
        # train(save_type, {"model_name":model_name, "config":{"n_components": 2, "perplexity":100,"n_jobs":8}}, sk_train_param)
        # visualise_latent_space(**eval_param)
        

        ae_training_param={"train_paths":[
            #"../../mtd_defence/datasets/uq/benign/Cam_1.csv"
            f"../data/{dataset}_train{'_filtered_0.2' if filtered else ''}.csv"
            ],
                "dataset":dataset,
                "epochs":100,
                "save_epochs":[1,20,40,60,80,100],
                "batch_size":1024,
                "shuffle":True, 
                "continue_training":False,
                "test_iter":100,
                "test_paths": {
                "ACK": "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding.csv",
                "SYN": "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_SYN_Flooding.csv",
                "UDP": "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_UDP_Flooding.csv"
                
                }
            }

        # fc = train_feature_cluster(ae_training_param["train_paths"][0], 2**10)

        
        
        with open("configs/nids_models.json","r") as f:
            nids_db=json.load(f)
            
        for dn, reduce_type, optimizer, act, lat in product( denoise, reduce_types.items(), optimizers.items(), activation, lat_dims):
            reduce_name, reduce = reduce_type
            opt_name, opt_dict = optimizer
            # model_name = f"{scaler_type}_{reduce_name}_{alpha}_{'_'.join(loss)}_{opt_name}{'_denoising' if dn else ''}_2d_{lr}{'_double_recon' if dr else ''}"
            model_name=f"{'denoising_' if dn else ''}autoencoder_{act}_{lat}{'_filtered_0.2' if filtered else ''}{'_D' if loss==['recon_dist_loss'] else ''}"
            opt = tf.keras.optimizers.get(opt_dict)
            print(model_name)

            ae_param = {"input_dim": 100,
                        "latent_dim": lat,
                        "latent_slices": 200,
                        "window_size": 64,
                        "step_size": 1,
                        "batch_size": 2**7,
                        "fc": None,
                        "activation":act,
                        "reduce": reduce,
                        "alpha": alpha,
                        "include_losses": loss,
                        "opt": opt,
                        # "scaler": None
                        "scaler": tf_scaler,
                        "num_neurons": [50],
                        "denoise": dn,
                        "shape":"circular",
                        "double_recon": False,
                        "model_name":model_name
                        }

            
            
            print(f"training {model_name}")
            train("FC", ae_param, ae_training_param)
            
            
            for i in ae_training_param["save_epochs"]:
                nids_db[f"{dataset}_{model_name}_{i}"]={
                    "abbrev":f"{'DAE' if dn else 'AE'}{'_R' if act=='relu' else ''}{'_25'if lat==25 else ''}{'F0.2' if filtered else ''}{'D' if loss==['recon_dist_loss'] else ''}-{i}",
                    "path":f"../models/{dataset}/{model_name}_{i}",
                        "save_type": "tf",
            "func_name": "call",
            "dr_output_index": 0,
            "ad_output_index": 1,
            "dtype": "float32"}
                
            with open("configs/nids_models.json","w") as f:    
                json.dump(nids_db, f, indent=4)
            

                
            # visualise_latent_space(**eval_param)
