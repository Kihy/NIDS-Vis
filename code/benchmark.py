import tensorflow as tf
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import pickle


def get_map_func(scaler):
    data_min = scaler.data_min_
    data_range = scaler.data_range_

    def parse_csv(line):
        line = tf.io.decode_csv(line, record_defaults=[
                                0.0 for i in range(100)], select_cols=list(range(100)))
        data = tf.transpose(tf.stack(line))
        data = (data-data_min) / data_range
        return data
    return parse_csv


def benchmark(ds, epochs=1):
    start_time = time.perf_counter()
    for epoch_num in range(epochs):
        for i in tqdm(ds):
            # df = i.sample(frac=1)
            # print(i.shape)
            time.sleep(0.001)
    print("Execution time:", time.perf_counter() - start_time)


scaler_path = "../../mtd_defence/models/uq/autoencoder/Cam_1_scaler.pkl"
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
path = "../../mtd_defence/datasets/uq/benign/Cam_1.csv"
traffic_ds = pd.read_csv(path, chunksize=1024,
                         usecols=list(range(100)), header=None, skiprows=1, dtype=np.float32)
tf_ds = tf.data.TextLineDataset(path)
tf_ds = tf_ds.skip(1).shuffle(
    1024*300, reshuffle_each_iteration=True).batch(1024).filter(lambda x: tf.random.uniform(shape=[]) < 0.5).map(get_map_func(scaler))
# benchmark(traffic_ds)
benchmark(tf_ds, 2)
# benchmark(tf.data.Dataset.range(1).interleave(
#         lambda _: tf_ds,
#         num_parallel_calls=tf.data.AUTOTUNE
#     ))
