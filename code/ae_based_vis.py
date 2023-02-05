import tensorflow as tf
import numpy as np
import pickle
import plotly.graph_objects as go
from tqdm import tqdm
from experiment import get_latent_position
from helper import *

baseline_kitsune = "../../mtd_defence/models/uq/kitsune/Cam_1.pkl"

with open(baseline_kitsune, "rb") as m:
    baseline_kitsune = pickle.load(m)


ae_name = "min_max_mean_10.0_recon_loss_adam_denoising_2d_0.001"
ae_path = f"../models/{ae_name}"
autoencoder = tf.keras.models.load_model(ae_path)


scaler_path = "../../mtd_defence/models/uq/autoencoder/Cam_1_min_max_scaler.pkl"
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

bottom_left = [-7.0,-7.0]
top_right = [7.0, 7.0]
resolution = [400, 400]

grid_points_x = tf.linspace(
    bottom_left[0], top_right[0], resolution[0])
grid_points_y = tf.linspace(
    bottom_left[1], top_right[1], resolution[1])
xv, yv = tf.meshgrid(grid_points_x, grid_points_y, indexing='xy')
xi, yi = np.indices((resolution[0], resolution[1]), dtype=np.int64)

# calculate corresponding decoded value
# coords of shape resolution * resolution, 2
coords = tf.reshape(
    tf.concat((xv[:, :, tf.newaxis], yv[:, :, tf.newaxis]), axis=-1), [-1, 2])

#process in batches
batch_size = 2**13
num_coords = coords.shape[0]
split_sizes = [batch_size] * \
    (num_coords // batch_size) + [num_coords % batch_size]

kitsune_score = []
test_batch = []
for batch in tqdm(tf.split(coords, split_sizes)):
    decoded_samples = autoencoder.postprocess(autoencoder.decoder(batch))
    decoded_samples = scaler.inverse_transform(decoded_samples)

    kitsune_score.append(baseline_kitsune.predict(decoded_samples.numpy()))
    test_batch.append(batch)

kitsune_score = tf.concat(kitsune_score, 0)
kitsune_score = tf.reshape(kitsune_score, [resolution[0], resolution[1]])

threshold = 0.28151878818499115

# kitsune_score = tf.where(tf.less_equal(
#     kitsune_score, threshold), x=float('NaN'), y=kitsune_score)

contour = go.Contour(x=grid_points_x, y=grid_points_y,
                     z=kitsune_score,
                     name="Kitsune Decision Function",
                     text=kitsune_score,
                     line_smoothing=0.85,
                     contours_showlines=False,
                     colorbar=dict(x=1.2),
                     colorscale="Brwnyl",
                     opacity=1,
                     )

fig = go.Figure()
fig.add_trace(contour)
#
plots={"ACK_ori":{"path":"../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding.csv","frac":0.01, "color":"amp"},
"ACK_adv":{"path":"../../mtd_defence/datasets/uq/adversarial/Cam_1/ACK_Flooding/autoencoder_0.1_10_3_False_pso0.5/csv/Cam_1_ACK_Flooding_iter_0.csv","frac":0.1, "color":"Peach"},
"benign":{"path":"../../mtd_defence/datasets/uq/benign/Cam_1.csv","frac":0.05, "color":"Emrld"},
}


for name, config in plots.items():
    latent, recon, _ = get_latent_position(
        autoencoder, None, config["path"], frac=config["frac"], seed=0, dtype="float64")
    _, kit_recon, _ = get_latent_position(
        baseline_kitsune, None, config["path"], frac=config["frac"], seed=0, dtype="float64")

    fig.add_trace(go.Scattergl(x=latent[:, 0], y=latent[:, 1],
                               mode='markers',
                               opacity=0.3,
                               name=name,
                               text=[f"index: {i}\nkit_recon:{kit_recon[i]:.3f}\nae_recon:{recon[i]:.3f}" for i in range(
                                   latent.shape[0])],

                               marker=dict(size=10,
                                           colorscale=config["color"],
                                           # set color equal to a variable
                                           color=recon)
                               ))

fig.write_html(f"exp_figs/{ae_name}.html")
