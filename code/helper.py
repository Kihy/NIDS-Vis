import numpy as np
from sklearn.datasets import make_gaussian_quantiles, make_circles
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from numpy.random import default_rng
import tensorflow as tf
import subprocess as sp
from scipy.cluster.hierarchy import linkage, to_tree
import os
import tensorboard as tb
import json
import pandas as pd
from collections import defaultdict
import plotly.express as px
import pickle
import matplotlib.pyplot as plt
import json
from xpysom import XPySom
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
rng = default_rng()


def sample_d_sphere(b, n, d):
    u = np.random.normal(0,1,(b,n,d))  
    d=np.sum(u**2, axis=-1,keepdims=True) **(0.5)
    coord=u/d
    return coord

class SOM():
    def __init__(self, distance, n_func, size, sigma, learning_rate):
        self.som = XPySom(size, size, 100, sigma=sigma, learning_rate=learning_rate,
                           neighborhood_function=n_func, activation_distance=distance)

    def fit(self, x):
        self.som.train(x,1)

    def predict(self, x):
        return self.decision_function(x)

    def score_samples(self, x):
        return self.decision_function(x)

    def decision_function(self, x):
        return np.linalg.norm(self.som.quantization(x) - x, axis=1)

    def process(self, x):
        return self.decision_function([x])[0]

def load_model(save_type, path):
    if save_type=="tf":
        model = tf.keras.models.load_model(path)
    elif save_type=="pkl":
        with open(path, "rb") as f:
            model = pickle.load(f)
    else:
        raise ValueError("unknown save_type type")
    return model 

class BaseModel:
    def __init__(self, name, save_type, path, func_name, ad_output_index = None, 
                 dr_output_index=None, threshold=None, flip_score=False, scaler=None, dtype="float64", min_feature=None,abbrev=None, **kwargs):
        model=load_model(save_type, path)
        self.pred_func=getattr(model, func_name)
        # print(name, model.summary())
        if hasattr(model, "inverse_transform"):
            self.inverse_func=model.inverse_transform
        else:
            self.inverse_func=None
        if isinstance(scaler, str):
            with open(scaler, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler=scaler         
        self.save_type=save_type
        self.ad_output_index=ad_output_index
        self.dr_output_index=dr_output_index
        self.flip_score=flip_score
        self.name=name
        self.threshold=threshold
        self.dtype=dtype
        self.min_feature=min_feature
        self.abbrev=abbrev


class DimensionalityReductionMixin:
    def transform(self, x):
        if self.scaler:
            x=self.scaler.transform(x)
        
        x=x.astype(self.dtype)
        
        latent_dim = self.pred_func(x)
        
        if self.dr_output_index is not None:
            latent_dim=latent_dim[self.dr_output_index]

        if isinstance(latent_dim,tf.Tensor):
            latent_dim=latent_dim.numpy()
        
        return latent_dim

class AnomalyDetectionMixin:
    def predict(self, x):
        if self.scaler:
            x=self.scaler.transform(x)
            
        x=x.astype(self.dtype)
        
        scores=self.pred_func(x)
        # if inverse transformation is available, it is a dimensionality reduction model
        if self.inverse_func:
            decoded=self.inverse_func(scores)
            scores=((x - decoded)**2).mean(axis=1)
            
        if self.ad_output_index is not None:
            scores=scores[self.ad_output_index]
        
        # some models have negative scores
        if self.flip_score:
            scores = -scores 
        
        if isinstance(scores,tf.Tensor):
            scores=scores.numpy()
        
        return scores 
    def decision(self, x, return_score=False):
        if self.threshold is None:
            raise AttributeError("No threshold set, only predict() is available")
        scores=self.predict(x)
        if return_score:
            return scores>self.threshold, scores
        else:
            return scores>self.threshold
        
class GenericDRModel(BaseModel, DimensionalityReductionMixin):
    pass

class GenericADModel(BaseModel, AnomalyDetectionMixin):
    pass 

class HybridModel(BaseModel, AnomalyDetectionMixin, DimensionalityReductionMixin):
    pass
        

def get_map_func(scaler, ndim, dtype="float32"):
    # if scaler:
    #     data_min = scaler.data_min_
    #     data_range = scaler.data_range_
    #
    # else:
    #     data_min = 0
    #     data_range = 1

    def parse_csv(line):
        
        line = tf.io.decode_csv(line, record_defaults=[
                                0. for i in range(ndim)], select_cols=list(range(ndim)))
        data = tf.transpose(tf.stack(tf.cast(line, dtype=dtype)))

        if scaler:
            data = scaler.transform(data)
        return data
    return parse_csv

def get_nids_model(model_id, threshold_key="100", load=True):
    with open("configs/nids_models.json") as f:
        nids_db=json.load(f)
    if "thresholds" in nids_db[model_id].keys():
        threshold=nids_db[model_id]["thresholds"][threshold_key]
    else:
        threshold=None
    if load:
        return GenericADModel(model_id,  threshold=threshold,**nids_db[model_id])
    else:
        ret_dict=dict(nids_db[model_id])
        ret_dict["threshold"]=threshold
        ret_dict["name"]=model_id
        return ret_dict

def get_files(file_ids):
    result=[]
    with open("configs/files.json") as f:
        file_db=json.load(f)
        for file_id in file_ids:
            file_db[file_id]["name"]=file_id
            result.append(file_db[file_id])
    return result
    
    

def get_dataset(path, batch_size, ndim=100, shuffle=False, scaler=None, frac=1, total_rows=None, read_with="tf", seed=None, drop_reminder=True, dtype="float32", skip_header=True):

    if read_with == "tf":
        tf_ds = tf.data.TextLineDataset(path)
        if total_rows:
            tf_ds=tf_ds.take(total_rows)
        if skip_header:
            tf_ds = tf_ds.skip(1)

        if shuffle:
            tf_ds = tf_ds.shuffle(
                1024 * 300, reshuffle_each_iteration=True, seed=seed)

        tf_ds = tf_ds.batch(batch_size, drop_remainder=drop_reminder)

        if frac != 1:
            tf_ds = tf_ds.filter(
                lambda x: tf.random.uniform(shape=[], seed=seed) < frac)
           
        tf_ds = tf_ds.map(get_map_func(scaler, ndim, dtype=dtype))

    elif read_with == "pd":
        if seed:
            np.random.seed(seed)
        pd_data = pd.read_csv(path, usecols=list(
            range(100)), header=None, skiprows=lambda x: x == 0 or np.random.rand() > frac, dtype=dtype)
        if scaler:
            pd_data = scaler.transform(pd_data)
            
        if shuffle:
            pd_data = pd_data.sample(frac=1).reset_index(drop=True)


        if batch_size>0:
            tf_ds = tf.data.Dataset.from_tensor_slices(pd_data)
            tf_ds = tf_ds.batch(batch_size, drop_remainder=True)
        else:
            tf_ds=pd_data

    else:
        raise ValueError("Unknown read type")

    return tf_ds


def make_spiral(n_points, noise=0.1, scale=10):
    n = np.sqrt(rng.random((n_points, 1))) * 720 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + rng.random((n_points, 1)) * noise
    d1y = np.sin(n) * n + rng.random((n_points, 1)) * noise
    d1x /= scale
    d1y /= scale
    return np.hstack((d1x, d1y))


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


def get_tensorboard_scalar(experiment_id):
    # = "XDAoZmabTQ2GWZ6axpwYlA"
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    df = df.dropna()

    df = df[(df["step"] == 1000) & df["tag"].str.contains(
        "activations|linearity|weights|MDR") & df["run"].str.contains("sw_loss|distance_loss")]

    df = df.pivot(index="run", columns="tag", values="value")

    df["average_linearity"] = (
        df["linearity/ACK"] + df["linearity/SYN"] + df["linearity/UDP"]) / 3

    df["average_MDR"] = (
        df["MDR/ACK"] + df["MDR/SYN"] + df["MDR/UDP"]) / 3

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
                                               window_size + 1, delta=step_size), 0))
    )
    return tf.gather(data, window_map)


def unslide_window(time_series, window_size, step_size, original_length):

    window_map = (
        # expand_dims are used to convert a 1D array to 2D array.
        tf.expand_dims(tf.range(window_size), 0)
        + tf.transpose(tf.expand_dims(tf.range(original_length - \
                                               window_size + 1, delta=step_size), 0))
    )

    # count occurrences of index
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
        # parameter:
        self.n = n
        # varaibles
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

        self.c = self.N / (self.N + n) * tmp + n / (self.N + n) * newmean
        self.std = self.N / (self.N + n) * self.std**2 + n / (self.N + n) * newstd**2 +\
            self.N * n / (self.N + n)**2 * (tmp - newmean)**2
        self.std = np.sqrt(self.std)

        self.N += n

        c_rs = (self.std**2 * self.N)
        self.c_rs += c_rs
        c_rt = c_rs**0.5
        self.C += np.outer(c_rt, c_rt)
    # creates the current correlation distance matrix between the features

    def corrDist(self):
        c_rs_sqrt = np.sqrt(self.c_rs)
        C_rs_sqrt = np.outer(c_rs_sqrt, c_rs_sqrt)
        # this protects against dive by zero erros (occurs when a feature is a constant)
        C_rs_sqrt[C_rs_sqrt == 0] = 1e-100
        D = 1 - self.C / C_rs_sqrt  # the correlation distance matrix
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


@tf.function
def point_line_distance(lines, points):
    t = tf.einsum('ik,jk->ji', lines, points) / \
        tf.einsum('ij,ij->i', lines, lines)[tf.newaxis, :]
    dist = points[:, tf.newaxis, :] - tf.einsum('ij,jk->ijk', t, lines)
    dist = tf.linalg.norm(dist, axis=-1)
    return dist


def normalize_attack_data(atk_path):
    scaler = MinMaxScaler()
    data = pd.read_csv(atk_path, chunksize=1024, usecols=list(
        range(100)), header=None, skiprows=1, dtype=np.float32)

    for chunk in tqdm(data):
        scaler = scaler.partial_fit(chunk)

    data = pd.read_csv(atk_path, chunksize=1024, usecols=list(
        range(100)), header=None, skiprows=1, dtype=np.float32)

    with open(atk_path[:-4] + "_scaled.csv", "a") as csvfile:
        for chunk in tqdm(data):
            scaled = scaler.transform(chunk)

            np.savetxt(csvfile, scaled, delimiter=',')
    print(f"done{atk_path}")


class LogMinMaxScaler:
    def __init__(self, dtype="float32"):
        self.data_min_ = 0
        self.data_max_ = 0
        self.first_fit = True
        self.dtype = dtype

    def partial_fit(self, X):
        X = bi_symlog(X)
        data_min = tf.reduce_min(X, axis=0)
        data_max = tf.reduce_max(X, axis=0)
        if self.first_fit:
            self.data_min_ = data_min
            self.data_max_ = data_max
            self.first_fit = False
        else:
            self.data_min_ = tf.math.minimum(data_min, self.data_min_)
            self.data_max_ = tf.math.maximum(data_max, self.data_max_)

    @tf.function
    def transform(self, X):
        # return X
        X = bi_symlog(X)

        return (X - self.data_min_) / (self.data_max_ - self.data_min_)

    @tf.function
    def inverse_transform(self, X):
        # return X
        X = X * (self.data_max_ - self.data_min_) + self.data_min_
        return bi_symexp(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def fit(self, X):
        X = bi_symlog(x)
        self.data_min_ = tf.reduce_min(X, axis=0)
        self.data_max_ = tf.reduce_max(X, axis=0)

    def get_config(self):
        return {"data_min_": self.data_min_.numpy(),
                "data_max_": self.data_max_.numpy(),
                "first_fit": self.first_fit,
                "dtype": self.dtype}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.function
def bi_symexp(x):
    c = tf.math.divide(tf.constant(1., dtype=x.dtype),
                       tf.math.log(tf.constant(10., dtype=x.dtype)))

    return tf.math.sign(x) * c * (10.**tf.math.abs(x) - 1.)


@tf.function
def bi_symlog(x):
    c = tf.math.divide(tf.constant(1., dtype=x.dtype),
                       tf.math.log(tf.constant(10., dtype=x.dtype)))
    return tf.math.sign(x) * tf.math.log1p(tf.math.abs(x / c)) / tf.math.log(tf.constant(10., dtype=x.dtype))


@tf.function
def ph_calc(matrix):
    n_vertices = matrix.shape[0]
    parents = tf.range(n_vertices, dtype=tf.int64)

    upper_triangle = tf.linalg.band_part(matrix, 0, -1)
    triu_indices = tf.where(upper_triangle != 0.)

    edge_weights = tf.gather_nd(matrix, triu_indices)
    edge_indices = tf.argsort(edge_weights)

    # 1st dimension: 'source' vertex index of edge
    # 2nd dimension: 'target' vertex index of edge
    persistence_pairs = tf.zeros([n_vertices - 1, 2], dtype=tf.int64)
    pair_idx = 0
    for edge_index in edge_indices:

        edge_weight = edge_weights[edge_index]
        u = tf.gather_nd(triu_indices, [edge_index, 0])
        v = tf.gather_nd(triu_indices, [edge_index, 1])

        u_root = u
        v_root = v
        while tf.gather(parents, u_root) != u_root:
            u_root = tf.gather(parents, u_root)
        while tf.gather(parents, v_root) != v_root:
            v_root = tf.gather(parents, v_root)

        # Not an edge of the MST, so skip it
        if u_root == v_root:
            continue

        if u != v:
            if u_root > v_root:
                parents = tf.tensor_scatter_nd_update(
                    parents, [[v_root]], [u_root])
            if u_root < v_root:
                parents = tf.tensor_scatter_nd_update(
                    parents, [[u_root]], [v_root])

        if u < v:
            persistence_pairs = tf.tensor_scatter_nd_update(
                persistence_pairs, [[pair_idx]], [[u, v]])

            pair_idx += 1
        else:
            persistence_pairs = tf.tensor_scatter_nd_update(
                persistence_pairs, [[pair_idx]], [[v, u]])
            pair_idx += 1
        if pair_idx == n_vertices - 1:
            break

    return persistence_pairs


def count_matching_pairs(pairs1, pairs2):
    a = tf.sparse.to_dense(pairs1) * tf.sparse.to_dense(pairs2)
    return tf.reduce_sum(a)


@tf.function
def pairwise_distance(x):
    x_flat = tf.reshape(x, [x.shape[0], -1])
    return tf.norm(x_flat[:, None] - x_flat, ord='euclidean', axis=-1)


def select_distances_from_pairs(distance_matrix, pairs):
    selected_distances = tf.gather_nd(distance_matrix, pairs)
    return selected_distances


@tf.function
def sig_error(signature1, signature2):
    """Compute distance between two topological signatures."""
    d0_death_diff = tf.keras.losses.mean_squared_error(
        signature1 / tf.reduce_max(signature1), signature2 / tf.reduce_max(signature2))

    # d1_death_diff = tf.keras.losses.mean_squared_error(
    #     signature1[1]["death"] / s1_dim, signature2[1]["death"] / s2_dim)
    # d1_birth_diff = tf.keras.losses.mean_squared_error(
    #     signature1[1]["birth"] / s1_dim, signature2[1]["birth"] / s2_dim)

    return d0_death_diff  # + d1_death_diff + d1_birth_diff


def get_gpu_memory():
    def _output_to_list(x): return x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0])
                          for i, x in enumerate(memory_free_info)]
    print(memory_free_values)

def divide_no_nan(a,b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
class MinMaxScaler:
    def __init__(self, dtype="float32"):
        self.data_min_ = 0
        self.data_max_ = 0
        self.first_fit = True
        self.dtype = dtype

    def partial_fit(self, X):
        data_min = tf.reduce_min(X, axis=0)
        data_max = tf.reduce_max(X, axis=0)
        if self.first_fit:
            self.data_min_ = data_min
            self.data_max_ = data_max
            self.first_fit = False
        else:
            self.data_min_ = tf.math.minimum(data_min, self.data_min_)
            self.data_max_ = tf.math.maximum(data_max, self.data_max_)

    @tf.function
    def transform(self, X):
        original_dtype=X.dtype
        X=tf.cast(X, self.dtype)
        transformed_value= tf.math.divide_no_nan((X - self.data_min_) ,(self.data_max_ - self.data_min_))
        return tf.cast(transformed_value, original_dtype)

    @tf.function
    def inverse_transform(self, X):
        original_dtype=X.dtype
        X=tf.cast(X, self.dtype)
        transformed_value= X * (self.data_max_ - self.data_min_) + self.data_min_
        return tf.cast(transformed_value, original_dtype)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def fit(self, X):
        self.data_min_ = tf.reduce_min(X, axis=0)
        self.data_max_ = tf.reduce_max(X, axis=0)

    def get_config(self):
        return {"data_min_": self.data_min_.numpy(),
                "data_max_": self.data_max_.numpy(),
                "first_fit": self.first_fit,
                "dtype": self.dtype}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def save_minmax_scaler(benign_path, save_path, type="normal", dtype="float32"):
    if type == "log":
        scaler = LogMinMaxScaler(dtype=dtype)
    elif type == "normal":
        scaler = MinMaxScaler(dtype=dtype)
    data = pd.read_csv(benign_path, chunksize=1024, usecols=list(
        range(100)), header=None, skiprows=1, dtype=dtype)

    for chunk in tqdm(data):
        scaler.partial_fit(chunk)

    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)


def gradient_estimate(func, x_t, delta_t, direction=None,feature_range=None):
    if x_t.ndim == 1:
        x_t = np.expand_dims(x_t, axis=0)

    if feature_range is not None:
        delta=delta_t*feature_range
    else:
        delta=delta_t

    if direction is not None:
        direction/=np.linalg.norm(direction, axis=1,keepdims=True)
        h=direction*delta
        forward_samples=x_t+h 
        backward_samples=x_t-h
    else:

        h=np.identity(x_t.shape[1])*delta
        
        forward_samples=x_t[:,np.newaxis,:]+h[np.newaxis,:,:]
        forward_samples=forward_samples.reshape((-1,x_t.shape[1]))
        
        backward_samples=x_t[:,np.newaxis,:]-h[np.newaxis,:,:]
        backward_samples=backward_samples.reshape((-1,x_t.shape[1]))
    
    forward_out=func(forward_samples)
    backward_out=func(backward_samples)
                           
    if backward_out.ndim==1:
        n_output=1
        backward_out=backward_out.reshape(x_t.shape[0],-1)
        forward_out=forward_out.reshape([x_t.shape[0],-1])
    else:
        n_output=backward_out.shape[1]
        backward_out=backward_out.reshape(x_t.shape[0],-1,n_output)
        forward_out=forward_out.reshape((x_t.shape[0],-1,n_output))
    
    gradient=(forward_out-backward_out)/(2*delta_t)

    #transpose to make it compatible with tf.gradient_tape.batch_jacobian
    if n_output>1:
        gradient=np.transpose(gradient, axes=[0,2,1])

    return gradient

def monte_carlo_estimate(func, x_t, B_t, delta_t, logger=None):
    """estimate the gradient of func at x_t. Estimated gradient is not scaled

    Args:
        func (function): function to evaluate
        x_t (ndarray): array of input
        B_t (int): number of random vectors
        delta_t (float): radius of hypersphere to sample from
        feature_range (ndarray): feature range used to proportionally sample the input feature space
        logger (logger, optional): logger used to log info. Defaults to None.

    Raises:
        Exception: if delta_t is too small and all samples have same value

    Returns:
        ndarray: the estimated gradient
    """
    # print("MC delta_t", delta_t)
    if x_t.ndim == 1:
        x_t = np.expand_dims(x_t, axis=0)

    u = rng.normal(0, 1, (B_t * x_t.shape[0], x_t.shape[1]))
    d = np.sum(u**2, axis=1, keepdims=True)**0.5
    ub = u / d
    ub = np.reshape(ub, (x_t.shape[0], B_t, x_t.shape[1]))

    # logger.info(f"x_t {x_t.shape} delta_t {delta_t.shape}, feautre_range {feature_range.shape}")
    random_samples = x_t[ :,np.newaxis, :] + \
        delta_t * ub

    random_samples = random_samples.reshape((-1, x_t.shape[1]))

    inner_result = func(random_samples)

    inner_result = inner_result.reshape(-1, B_t)

    baseline = np.mean(inner_result, axis=1, keepdims=True)

    print(baseline.shape, inner_result.shape)

    gradient_direction = np.einsum("ij,ijk->ik", (inner_result - baseline), ub) 
    # logger.info(f"random samples {random_samples.shape} inner result {inner_result.shape} baseline {baseline.shape} ub {ub.shape}")
    # logger.info(f"input {x_t.shape} gradient_direction {gradient_direction.shape}" )
    
    if np.any(np.sum(gradient_direction, axis=1) == 0):
        # logger.info(f"unbiased_estimate {unbiased_estimate}")
        logger.warning(f"baseline {baseline}")
        logger.warning(f"delta_t {delta_t}")
        score = func(random_samples)
        score2 = func(x_t)
        logger.warning(f"original score {score2}")
        score = score.reshape(-1, B_t)
        logger.warning(f"random sample score {score}")
        raise Exception()

    return gradient_direction.astype(x_t.dtype)

def get_latent_position(model, scaler, path, batch_size=1024, frac=1, seed=None, shuffle=True,
                        dtype="float32", read_with="tf", skip_header=True):

    traffic_ds = get_dataset(path, batch_size,
                             scaler=scaler, frac=frac, read_with=read_with, dtype=dtype, seed=seed, skip_header=skip_header, shuffle=shuffle)


    latent_dim = []
    recon_array = []
    
    processed = 0
    # Test autoencoder
    for data in tqdm(traffic_ds, leave=False, desc=f"Visualize: {path}"):
        processed += data.shape[0]

        if isinstance(model, AnomalyDetectionMixin):
            score = model.predict(data.numpy())
            recon_array.append(score)
            
        if isinstance(model, DimensionalityReductionMixin):
            encoded = model.transform(data.numpy())
            latent_dim.append(encoded)
        
    print(f"{path} processed {processed} packets")
    
    if len(latent_dim)>0:
        latent_dim=np.vstack(latent_dim)
    if len(recon_array)>0:
        recon_array=np.hstack(recon_array)
    
    return latent_dim, recon_array


def plot_hist(df_path):
    data = pd.read_csv(df_path, usecols=list(
        range(100)), header=None, skiprows=1, dtype=np.float32)

    # data = bi_symlog(data)
    ax = data.hist(bins=20, figsize=(15, 15), sharey=True)
    fig = ax[0][0].get_figure()

    fig.tight_layout()
    plt.ticklabel_format(axis="both", style="scientific", useOffset=False)

    fig.savefig("exp_figs/benign_hist.png")


if __name__ == '__main__':
    # get_tensorboard_scalar("tr8Q0AknR9eRSjy5GWk5bA")
    # plot_hist("../../mtd_defence/datasets/uq/benign/Cam_1.csv")
    save_minmax_scaler("../../mtd_defence/datasets/uq/benign/Cam_1.csv",
                       "../../mtd_defence/models/uq/autoencoder/Cam_1_min_max_scaler.pkl",
                       type="normal", dtype="float32")
    # normalize_attack_data(
    #     "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding.csv")
