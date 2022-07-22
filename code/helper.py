import numpy as np
from sklearn.datasets import make_gaussian_quantiles, make_circles
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from numpy.random import default_rng
import tensorflow as tf
from scipy.cluster.hierarchy import linkage, to_tree
import os
import tensorboard as tb
import pandas as pd
from collections import defaultdict
import plotly.express as px
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
rng = default_rng()


def get_map_func(scaler):
    if scaler:
        data_min = scaler.data_min_
        data_range = scaler.data_range_

    else:
        data_min = 0
        data_range = 1

    def parse_csv(line):
        line = tf.io.decode_csv(line, record_defaults=[
                                0.0 for i in range(100)], select_cols=list(range(100)))
        data = tf.transpose(tf.stack(line))
        data = (data-data_min) / data_range
        return data
    return parse_csv


def get_dataset(path, batch_size, shuffle=False, scaler=None, frac=1, read_with="tf"):
    if read_with == "tf":
        tf_ds = tf.data.TextLineDataset(path)
        tf_ds = tf_ds.skip(1)

        if shuffle:
            tf_ds = tf_ds.shuffle(
                1024*300, reshuffle_each_iteration=True)

        tf_ds = tf_ds.batch(batch_size, drop_remainder=True)

        if frac != 1:
            tf_ds = tf_ds.filter(
                lambda x: tf.random.uniform(shape=[]) < frac)

        tf_ds = tf_ds.map(get_map_func(scaler))

    elif read_with == "pd":
        pd_data = pd.read_csv(path, usecols=list(
            range(100)), header=None, skiprows=lambda x: x == 0 or np.random.rand() > frac, dtype=np.float32)
        if scaler:
            pd_data = scaler.transform(pd_data)
        tf_ds = tf.data.Dataset.from_tensor_slices(pd_data)
        tf_ds = tf_ds.batch(batch_size, drop_remainder=True)

    return tf_ds


def make_spiral(n_points, noise=0.1, scale=10):
    n = np.sqrt(rng.random((n_points, 1))) * 720 * (2*np.pi)/360
    d1x = -np.cos(n)*n + rng.random((n_points, 1)) * noise
    d1y = np.sin(n)*n + rng.random((n_points, 1)) * noise
    d1x /= scale
    d1y /= scale
    return np.hstack((d1x, d1y))


@tf.function
def generate_theta(n, dim):
    """generates n slices to be used to slice latent space, with dim dimensions. the slices are unit vectors"""
    theta, _ = tf.linalg.normalize(tf.random.normal(shape=[n, dim]), axis=1)
    return theta


@tf.function
def generate_z(n, dim, shape):
    """generates n samples with shape in dim dimensions, represents the prior distribution"""
    if shape == "uniform":
        z = 2*tf.random.uniform.uniform(shape=[n, dim])-1
    elif shape == "circular":
        R = 1
        theta = tf.random.uniform(shape=[n], maxval=2*np.pi)
        radius = tf.random.uniform(shape=[n], maxval=R) ** 0.5

        x = radius * tf.math.cos(theta)
        y = radius * tf.math.sin(theta)

        z = tf.transpose(tf.stack((x, y)))

    return z


def get_tensorboard_scalar(experiment_id):
    # = "XDAoZmabTQ2GWZ6axpwYlA"
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    df = df.dropna()

    df = df[(df["step"] == 1000) & df["tag"].str.contains(
        "activations|linearity|weights|MDR") & df["run"].str.contains("sw_loss|distance_loss")]

    df = df.pivot(index="run", columns="tag", values="value")

    df["average_linearity"] = (
        df["linearity/ACK"] + df["linearity/SYN"]+df["linearity/UDP"])/3

    df["average_MDR"] = (
        df["MDR/ACK"] + df["MDR/SYN"]+df["MDR/UDP"])/3

    def label_loss(row):
        run_name = row.name
        if "recon_loss_sw_loss" in run_name:
            return "sw_loss"
        if "recon_loss_distance_loss_sw_loss" in run_name:
            return "all"

    def label_act(row):
        run_name = row.name
        if "act_bn" in run_name:
            return "act_bn"
        if "act" in run_name:
            return "act"
        if "batch_norm" in run_name:
            return "batch_norm"
        return "normal"

    def label_run(row):
        run_name = row.name
        if "amsgrad" in run_name:
            return "amsgrad"
        if "rmsprop" in run_name:
            return "rmsprop"
        if "sgd" in run_name:
            return "sgd"
        if "adam" in run_name:
            return "adam"
    df["opt"] = df.apply(lambda row: label_run(row), axis=1)
    df["loss"] = df.apply(lambda row: label_loss(row), axis=1)
    df["aux"] = df.apply(lambda row: label_act(row), axis=1)

    fig = px.scatter(df, x="activations/benign",
                     y="average_linearity", color="aux", hover_data={df.index.name: (True, df.index)}, symbol="loss")
    fig.write_html("exp_figs/test.html")

    fig = px.scatter(df, x="weights",
                     y="average_linearity", color="aux", hover_data={df.index.name: (True, df.index)}, symbol="loss")
    fig.write_html("exp_figs/test2.html")


def generate_sliding_window(data, window_size, step_size=1):

    batch_size = tf.shape(data)[0]
    window_map = (
        # expand_dims are used to convert a 1D array to 2D array.
        tf.expand_dims(tf.range(window_size), 0)
        + tf.transpose(tf.expand_dims(tf.range(batch_size - \
                                               window_size+1, delta=step_size), 0))
    )
    return tf.gather(data, window_map)


def unslide_window(time_series, window_size, step_size, original_length):

    window_map = (
        # expand_dims are used to convert a 1D array to 2D array.
        tf.expand_dims(tf.range(window_size), 0)
        + tf.transpose(tf.expand_dims(tf.range(original_length - \
                                               window_size+1, delta=step_size), 0))
    )

    #count occurrences of index
    counts = tf.math.bincount(tf.reshape(window_map, [-1]), dtype=tf.float32)

    # calculate sum of the indices

    sums = tf.scatter_nd(tf.expand_dims(window_map, axis=-1),
                         time_series, tf.constant([original_length, 100]))

    recon = tf.math.divide(sums, tf.expand_dims(counts, axis=-1))
    return recon


def cluster_feature(data, fc):
    if fc is None:
        return data
    clustered = tf.gather(data, fc, axis=-1)
    return clustered


def uncluster_feature(data, fc):
    if fc is None:
        return data
    unshuf_order = np.zeros(100, dtype=np.int32)
    unshuf_order[fc] = np.arange(100, dtype=np.int32)
    # Unshuffle the shuffled data
    return tf.gather(data, unshuf_order, axis=-1)


class corClust:
    def __init__(self, n):
        #parameter:
        self.n = n
        #varaibles
        self.c = np.zeros(n)  # linear num of features
        self.c_rs = np.zeros(n)  # squared sum of feature residules
        self.C = np.zeros((n, n))  # partial correlation matrix
        self.N = 0  # number of updates performed
        self.std = np.zeros(n)

    # x: a numpy vector of length n

    def update(self, x):

        newmean = x.mean(axis=0)
        newstd = x.std(axis=0)

        n = x.shape[0]
        tmp = self.c

        self.c = self.N/(self.N+n)*tmp + n/(self.N+n)*newmean
        self.std = self.N/(self.N+n)*self.std**2 + n/(self.N+n)*newstd**2 +\
            self.N*n/(self.N+n)**2 * (tmp - newmean)**2
        self.std = np.sqrt(self.std)

        self.N += n

        c_rs = (self.std**2*self.N)
        self.c_rs += c_rs
        c_rt = c_rs**0.5
        self.C += np.outer(c_rt, c_rt)
    # creates the current correlation distance matrix between the features

    def corrDist(self):
        c_rs_sqrt = np.sqrt(self.c_rs)
        C_rs_sqrt = np.outer(c_rs_sqrt, c_rs_sqrt)
        # this protects against dive by zero erros (occurs when a feature is a constant)
        C_rs_sqrt[C_rs_sqrt == 0] = 1e-100
        D = 1-self.C/C_rs_sqrt  # the correlation distance matrix
        D[D < 0] = 0  # small negatives may appear due to the incremental fashion in which we update the mean. Therefore, we 'fix' them
        return D

    # clusters the features together, having no more than maxClust features per cluster
    def cluster(self, maxClust):
        D = self.corrDist()
        # create a linkage matrix based on the distance matrix
        Z = linkage(D[np.triu_indices(self.n, 1)])
        if maxClust < 1:
            maxClust = 1
        if maxClust > self.n:
            maxClust = self.n
        map = self.__breakClust__(to_tree(Z), maxClust)
        return map

    # a recursive helper function which breaks down the dendrogram branches until all clusters have no more than maxClust elements
    def __breakClust__(self, dendro, maxClust):
        if dendro.count <= maxClust:  # base case: we found a minimal cluster, so mark it
            # return the origional ids of the features in this cluster
            return [dendro.pre_order()]
        return self.__breakClust__(dendro.get_left(), maxClust) + self.__breakClust__(dendro.get_right(), maxClust)


def normalize_attack_data(atk_path):
    scaler = MinMaxScaler()
    data = pd.read_csv(atk_path, chunksize=1024, usecols=list(
        range(100)), header=None, skiprows=1, dtype=np.float32)

    for chunk in tqdm(data):
        scaler = scaler.partial_fit(chunk)

    data = pd.read_csv(atk_path, chunksize=1024, usecols=list(
        range(100)), header=None, skiprows=1, dtype=np.float32)

    with open(atk_path[:-4]+"_scaled.csv", "a") as csvfile:
        for chunk in tqdm(data):
            scaled = scaler.transform(chunk)

            np.savetxt(csvfile, scaled, delimiter=',')
    print(f"done{atk_path}")


if __name__ == '__main__':
    get_tensorboard_scalar("tr8Q0AknR9eRSjy5GWk5bA")
    # normalize_attack_data(
    #     "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding.csv")
