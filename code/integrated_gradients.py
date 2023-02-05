from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import pandas as pd
from numpy.random import default_rng
import numpy as np
import tensorflow as tf
import os
from keract import display_activations, get_activations
from tqdm import tqdm
from experiment import visualise_latent_space, test_ae, get_latent_position, plot_latent_similarity
import tensorflow_probability as tfp
import scipy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('WARNING')
rng = default_rng(0)


def l1_distance(x1, x2):
    """Returns L1 distance between two points."""
    return tf.reduce_sum(tf.abs(x1 - x2))


def translate_x_to_alpha(x, x_input, x_baseline):
    """Translates a point on straight-line path to its corresponding alpha value.
    Args:
    x: the point on the straight-line path.
    x_input: the end point of the straight-line path.
    x_baseline: the start point of the straight-line path.
    Returns:
    The alpha value in range [0, 1] that shows the relative location of the
    point between x_baseline and x_input.
    """

    return (x - x_baseline) / (x_input - x_baseline)


def translate_alpha_to_x(alpha, x_input, x_baseline):
    """Translates alpha to the point coordinates within straight-line interval.
    Args:
    alpha: the relative location of the point between x_baseline and x_input.
    x_input: the end point of the straight-line path.
    x_baseline: the start point of the straight-line path.
    Returns:
    The coordinates of the point within [x_baseline, x_input] interval
    that correspond to the given value of alpha.
    """
    return x_baseline + (x_input - x_baseline) * alpha


def guided_ig(x_input, x_baseline, grad_func, steps=200, fraction=0.25, max_dist=0.02):

    x = tf.identity(x_baseline)
    l1_total = l1_distance(x_input, x_baseline)
    attr = tf.zeros_like(x_input, dtype=tf.float32)

    # If the input is equal to the baseline then the attribution is zero.
    total_diff = x_input - x_baseline
    if tf.reduce_sum(tf.abs(total_diff)) < 1e-6:
        return attr

    latent_points = []
    # Iterate through every step.
    for step in tqdm(range(steps)):
        # Calculate gradients and make a copy.

        fraction_new = fraction
        grad_actual, latent = grad_func(x)
        latent_points.append(latent[0])
        grad = tf.identity(grad_actual)
        # Calculate current step alpha and the ranges of allowed values for this
        # step.
        alpha = (step + 1.0) / steps
        alpha_min = max(alpha - max_dist, 0.0)
        alpha_max = min(alpha + max_dist, 1.0)
        x_min = translate_alpha_to_x(alpha_min, x_input, x_baseline)
        x_max = translate_alpha_to_x(alpha_max, x_input, x_baseline)
        # The goal of every step is to reduce L1 distance to the input.
        # `l1_target` is the desired L1 distance after completion of this step.
        l1_target = l1_total * (1 - (step + 1) / steps)

        # Iterate until the desired L1 distance has been reached.
        gamma = tf.convert_to_tensor(np.inf)
        while gamma > 1.0:
            x_old = tf.identity(x)

            x_alpha = translate_x_to_alpha(x, x_input, x_baseline)
            # replace nan with max
            x_alpha = tf.where(tf.logical_not(
                tf.math.is_finite(x_alpha)), alpha_max, x_alpha)

            # All features that fell behind the [alpha_min, alpha_max] interval in
            # terms of alpha, should be assigned the x_min values.
            x = tf.where(x_alpha < alpha_min, x_min, x)

            # Calculate current L1 distance from the input.
            l1_current = l1_distance(x, x_input)
            # If the current L1 distance is close enough to the desired one then
            # update the attribution and proceed to the next step.
            if tf.abs(l1_target - l1_current) < 1e-9:
                attr += (x - x_old) * grad_actual
                print("close enough")
                break

            # Features that reached `x_max` should not be included in the selection.
            # Assign very high gradients to them so they are excluded.
            # grad[x == x_max] = np.inf
            grad = tf.where(x == x_max, np.inf, grad)

            # Select features with the lowest absolute gradient.
            threshold = tfp.stats.percentile(
                tf.abs(grad), tf.cast(fraction_new * 100, tf.int32), interpolation='lower')

            s = tf.where(tf.logical_and(tf.abs(grad)
                         <= threshold, grad != np.inf), 1., 0.)

            # Find by how much the L1 distance can be reduced by changing only the
            # selected features.

            l1_s = tf.reduce_sum(tf.abs(x - x_max) * s)

            # Calculate ratio `gamma` that show how much the selected features should
            # be changed toward `x_max` to close the gap between current L1 and target
            # L1.
            if l1_s > 0:
                gamma_new = (l1_current - l1_target) / l1_s
                if tf.abs(gamma_new - gamma) < 1e-9:
                    fraction_new += 0.01
                gamma = gamma_new
            else:
                gamma = np.inf

            if gamma > 1.0:
                # Gamma higher than 1.0 means that changing selected features is not
                # enough to close the gap. Therefore change them as much as possible to
                # stay in the valid range.
                x = tf.where(s == 1., x_max, x)
            else:
                x_new = translate_alpha_to_x(gamma, x_max, x)
                x = tf.where(s == 1., x_new, x)

            # Update attribution to reflect changes in `x`.
            attr += (x - x_old) * grad_actual
    latent_points = tf.stack(latent_points, axis=0)
    return tf.reduce_sum(attr, axis=0), latent_points


def compute_gradients(features, model, recon=False):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(features)
        latent, loss, decoded = model(features)
        # tf.print(loss)
    if recon:
        jacobian = tape.gradient(loss, features)
    else:
        jacobian = tape.batch_jacobian(latent, features)

    return jacobian, latent


def integral_approximation(gradients, positions, reduce_points):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / \
        tf.constant(2.0, dtype=gradients.dtype)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)

    if reduce_points:
        positions = tf.gather(positions, [0, positions.shape[0] - 1], axis=0)
        diff = positions[1] - positions[0]
        slope = tf.math.atan2(diff[:, 1], diff[:, 0])
        return integrated_gradients, positions[1], slope
    else:
        positions = tf.squeeze(positions)
        return integrated_gradients, positions, None


@tf.function(reduce_retracing=True)
def one_batch(model, baseline, feature, alphas, batch_size, recon=False):

    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_feature(baseline=baseline,
                                                        feature=feature,
                                                        alphas=alphas)

    # Compute gradients between model outputs and interpolated inputs.
    interpolated_path_input_batch = tf.reshape(
        interpolated_path_input_batch, shape=[-1, 100])

    gradient_batch, latent = compute_gradients(features=interpolated_path_input_batch,
                                               model=model, recon=recon)

    if recon:
        gradient_batch = tf.reshape(
            gradient_batch, shape=[-1, batch_size, 100])
    else:
        gradient_batch = tf.reshape(
            gradient_batch, shape=[-1, batch_size, 2, 100])

    latent = tf.reshape(latent, shape=[-1, batch_size, 2])

    return gradient_batch, latent


def interpolate_feature(baseline, feature, alphas=None):
    # Generate m_steps intervals for integral_approximation() below.
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(feature, axis=0)
    delta = input_x - baseline_x
    features = baseline_x + alphas_x * delta
    return features


def integrated_gradients(model, baseline, feature, m_steps=50, batch_size=32, recon=False, frac=1, reduce_points=False):
    # Generate alphas.
    alphas = tf.cast(tf.linspace(start=0.0, stop=1.0,
                     num=m_steps + 1), dtype=baseline.dtype)

    if frac != 1:
        rng = np.random.default_rng()
        sample_idx = rng.choice(
            baseline.shape[0] - 1, int(baseline.shape[0] * (1 - frac)), replace=False)

        sample_idx = np.sort(sample_idx)

        baseline = np.delete(baseline, sample_idx + 1, axis=0)
        feature = np.delete(feature, sample_idx, axis=0)

    # Collect gradients.
    gradient_batches = []
    latent_batches = []
    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        gradient_batch, latent = one_batch(model,
                                           baseline, feature, alpha_batch, batch_size=baseline.shape[0], recon=recon)

        gradient_batches.append(gradient_batch)

        latent_batches.append(latent)

    # Concatenate path gradients together row-wise into single tensor.

    total_gradients = tf.concat(gradient_batches, axis=0)

    latent_points = tf.concat(latent_batches, axis=0)

    # Integral approximation through averaging gradients.

    avg_gradients, latent_points, slope = integral_approximation(
        gradients=total_gradients, positions=latent_points, reduce_points=reduce_points)

    diff = (feature - baseline)
    if not recon:
        diff = diff[:, tf.newaxis, :]

    # Scale integrated gradients with respect to input.
    integrated_gradients = diff * avg_gradients

    return integrated_gradients, latent_points, slope


def draw_summary_plot(all_attributions, all_data, feature_names, output_path, positions, bin_by_positions=False, bins=[10, 10]):
    all_idx = np.arange(all_data.shape[0])
    # if bin by position then use x,y coordinates, else use slope
    if bin_by_positions:
        stat, edge_x, edge_y, bin_number = scipy.stats.binned_statistic_2d(
            positions[:, 0], positions[:, 1], None, statistic='count', bins=bins, expand_binnumbers=True)
        n_rows = bins[0]
        n_cols = bins[1]
        fig = make_subplots(rows=bins[0], cols=bins[1],
                            subplot_titles=[f"x:{edge_x[i]:.2f}~{edge_x[i+1]:.2f} <br> y:{edge_y[j]:.2f}~{edge_y[j+1]:.2f} <br> count: {stat[i][j]}" for i in range(
                                bins[0]) for j in range(bins[1])],
                            vertical_spacing=0.02,
                            )

    else:
        positions = np.degrees(positions)
        stat, edge, bin_number = scipy.stats.binned_statistic(
            positions, positions, statistic='count', bins=bins, range=(-180, 180))
        n_rows = int(np.ceil(np.sqrt(bins)))
        n_cols = bins // n_rows + 1
        fig = make_subplots(rows=n_rows, cols=n_cols,
                            subplot_titles=[f"angle:{edge[i]:.2f}~{edge[i+1]:.2f} <br> count: {stat[i]}<br>" for i in range(
                                bins)],
                            vertical_spacing=0.1,
                            )

    for bin_i in range(1, n_rows + 1):
        for bin_j in range(1, n_cols + 1):
            if bin_by_positions:
                idx = np.logical_and(
                    bin_number[0] == bin_i, bin_number[1] == bin_j)
            else:
                bin_idx = (bin_i - 1) * n_cols + bin_j
                if bin_idx > bins:
                    break
                idx = bin_number == bin_idx
            attributions = all_attributions[idx]
            data = all_data[idx]
            index = all_idx[idx]
            pos = positions[idx]

            if data.size == 0:
                continue
            indices = np.argsort(np.abs(attributions).sum(axis=0))

            indices = indices[::-1][:10]

            y = 0
            vals = []
            text = []
            for i in indices[::-1]:
                attribution_value = attributions[:, i]
                feature_value = data[:, i]

                if bin_i == 1 and bin_j == 1 and y == 0:
                    colorbar = dict(
                        thickness=20,
                        tickvals=[attribution_value.min(
                        ), attribution_value.max()],
                        ticktext=["Low", "High"]
                    )
                else:
                    colorbar = None
                fig.add_trace(
                    go.Scattergl(
                        x=attribution_value, y=[
                            y + 4 * rng.random() - 2 for _ in range(attribution_value.shape[0])], mode="markers",
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo="x+text+name",
                        hovertext=[
                            f"feature_val {feature_value[j]:.2f}\n idx:{index[j]}" for j in range(data.shape[0])],
                        name=feature_names[i], marker=dict(color=np.clip(feature_value, np.percentile(feature_value, 5), np.percentile(feature_value, 95)),
                                                           colorscale="bluered",
                                                           colorbar=colorbar)), row=bin_i, col=bin_j)
                vals.append(y)
                text.append(feature_names[i])
                y += 7
                fig.update_yaxes(tickmode='array',
                                 tickvals=vals,
                                 ticktext=text,
                                 automargin=True,
                                 row=bin_i, col=bin_j
                                 )
                # fig.add_annotation(xref="x domain", yref="y domain", x=0.5, y=1.05, showarrow=False,
                #                    text=f"x:{np.min(pos[:,0]):.2f} ~ {np.max(pos[:,0]):.2f}  y:{np.min(pos[:,1]):.2f} ~ {np.max(pos[:,1]):.2f}", row=bin_i, col=bin_j)
    # fig.update_xaxes(type="log")
    fig.update_layout(autosize=False,
                      width=n_cols * 650,
                      height=n_rows * 400
                      )

    fig.write_html(output_path)


def visualise_ae_weights(autoencoder, name):
    fig, ax = plt.subplots(4, 2, figsize=(12, 12), gridspec_kw={})
    layer_idx = 0
    for layer in autoencoder.encoder.layers:
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        w, b = weights
        pos = ax[layer_idx, 0].imshow(w.T, cmap="gray")
        fig.colorbar(pos, ax=ax[layer_idx, 0])

        layer_idx += 1

    layer_idx = 3
    for layer in autoencoder.decoder.layers:
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        w, b = weights
        pos = ax[layer_idx, 1].imshow(w, cmap="gray")
        fig.colorbar(pos, ax=ax[layer_idx, 1])
        layer_idx -= 1

    fig.tight_layout()
    fig.savefig(f"exp_figs/{name}_weights.png", dpi=200)

    raise


def explain_single_packet(baseline, target, autoencoder):
    if tf.rank(baseline) == 1:
        baseline = baseline[tf.newaxis, :]
    if tf.rank(target) == 1:
        target = target[tf.newaxis, :]
    attributions, latent_points, _ = integrated_gradients(
        autoencoder, baseline, target, m_steps=1280, recon=True, reduce_points=False)

    attributions = tf.reduce_sum(attributions, axis=0)
    return attributions, latent_points


def draw_force_plot(attributions, feature_names, outfile,  recons, encoded, name, latent, top_k=10):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    idx = np.argsort(np.abs(attributions))[::-1]
    top_idx = idx[:top_k]
    head_length = np.sum(attributions) * 0.01
    cum_sum = recons[0]
    y = 0
    y_labels = []
    for i in top_idx:
        ax[0].arrow(cum_sum, y, attributions[i], 0,
                    length_includes_head=True, head_length=head_length, head_width=0.1)

        if attributions[i] > 0.:
            ax[0].text(cum_sum, y + 0.1, f"{attributions[i]:.2f}")
        else:
            ax[0].text(cum_sum + attributions[i], y
                       + 0.1, f"{attributions[i]:.2f}")
        y_labels.append(feature_names[i])
        cum_sum += attributions[i]
        y += 1
    ax[0].arrow(cum_sum, y, np.sum(attributions[idx[top_k:]]), 0,
                length_includes_head=True, head_length=head_length, head_width=0.1)
    ax[0].text(cum_sum, y + 0.1, f"{np.sum(attributions[idx[top_k:]]):.2f}")

    offset = 1
    for i in range(len(recons)):
        ax[0].axvline(recons[i])
        ax[0].text(recons[i], top_k + offset,
                   f"{name[i]}:{recons[i]:.3f}", ha="center")
        offset += 0.5

    y_labels.append("Rest")
    ax[0].set_yticks(np.arange(top_k + 1), y_labels)

    # draw latent coordinates
    ax[1].scatter(latent[:, 0], latent[:, 1], c=np.arange(latent.shape[0]))
    for i in range(len(encoded)):
        ax[1].plot(encoded[i][0], encoded[i][1], "x")
        ax[1].annotate(name[i], encoded[i])

    fig.tight_layout()
    fig.savefig(outfile)


def explain_single_packet_along_path(attack_data, pkt_idx, autoencoder, frac=1):
    batch_size = 1024
    path_attributions = []
    latent = []
    for i in tqdm(tf.range(0, pkt_idx, batch_size)):
        to = tf.minimum(i + batch_size, pkt_idx)
        path_attrib, latent_points, _ = integrated_gradients(
            autoencoder, attack_data[i:to],
            attack_data[i + 1:to + 1], m_steps=1000,
            recon=True, frac=frac, reduce_points=True)

        path_attributions.append(path_attrib)
        latent.append(latent_points)

    path_attributions = tf.concat(path_attributions, axis=0)
    latent = tf.concat(latent, axis=0)

    path_attributions = tf.reduce_sum(path_attributions, axis=0)
    return path_attributions, latent


def get_gradient(model):
    def grad(features):
        features = tf.convert_to_tensor(features)
        if tf.rank(features) == 1:
            features = tf.expand_dims(features, axis=0)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(features)
            latent, loss, decoded = model(features)
        return tape.gradient(loss, features), latent
    return grad


def get_all_packet(baseline, target, autoencoder, batch_size=128, msteps=100):
    total_samples = target.shape[0]
    if tf.rank(baseline) == 1:
        baseline = baseline[tf.newaxis, :]
        baseline = tf.tile(baseline, [total_samples, 1])

    all_attributions = []
    latent_positions = []
    slopes = []
    for i in tqdm(tf.range(0, total_samples, batch_size)):
        to = tf.minimum(i + batch_size, total_samples)
        attributions, latent_points, slope = integrated_gradients(
            autoencoder, baseline[i:to], target[i:to], m_steps=msteps, recon=True, reduce_points=True)
        latent_positions.append(latent_points)
        all_attributions.append(attributions)
        slopes.append(slope)

    return tf.concat(all_attributions, axis=0), tf.concat(latent_positions, axis=0), tf.concat(slopes, axis=0)


def get_all_gradients(target, autoencoder, batch_size=1024):
    total_samples = target.shape[0]
    grad_func = get_gradient(autoencoder)
    all_grads = []
    for i in tqdm(tf.range(0, total_samples, batch_size)):
        to = tf.minimum(i + batch_size, total_samples)
        grad, _ = grad_func(target[i:to])
        all_grads.append(grad)
    return tf.concat(all_grads, axis=0)


if __name__ == '__main__':
    ae_type = "FC"
    shape = "circular"
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

    # ae_name = "FC_circular_mean_10.0_recon_loss_entropy_sw_loss_rmsprop_act_bn_0.1"
    ae_name = "min_max_mean_10.0_recon_loss_sw_loss_contractive_loss_adam_denoising_2d_0.001_double_recon"
    # ae_name = "FC_circular_mean_10.0_recon_loss_entropy_sw_loss_amsgrad_act_1.0"

    ae_path = f"../models/{ae_name}"

    mse = tf.keras.losses.MeanSquaredError()

    autoencoder = tf.keras.models.load_model(ae_path)

    scaler_path = "../../mtd_defence/models/uq/autoencoder/Cam_1_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    attack_path = "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding.csv"
    benign_path = "../../mtd_defence/datasets/uq/benign/Cam_1.csv"

    sample_percentage = 0.01
    np.random.seed(0)
    attack_data = pd.read_csv(attack_path, usecols=list(
        range(100)), header=None, skiprows=lambda x: x == 0 or np.random.rand() > sample_percentage, dtype=np.float32)
    attack_data = scaler.transform(attack_data)

    benign_data = pd.read_csv(benign_path, usecols=list(
        range(100)), header=None, skiprows=lambda x: x == 0 or np.random.rand() > sample_percentage, dtype=np.float32)

    benign_data = scaler.transform(benign_data)

    baseline = benign_data[-1]

    # target_idx = 8017
    #
    # missing_feature = "HT_MI_1_weight"
    # missing_idx = feature_names.index(missing_feature)
    #
    # exp_attack_data = attack_data[target_idx]
    #
    # test_attack_data = np.copy(exp_attack_data)
    # test_attack_data[missing_idx] = baseline[missing_idx]
    #
    # encoded, recons, _ = autoencoder(
    #     tf.stack([baseline, exp_attack_data, test_attack_data], axis=0))
    # recons = recons.numpy()
    # name = ["baseline", "target", "test"]
    #
    # attrib1, latent = explain_single_packet(
    #     baseline, exp_attack_data, autoencoder)
    # draw_force_plot(attrib1.numpy(), feature_names,
    #                 "exp_figs/single.png", recons, encoded, name, latent)
    # print(tf.reduce_sum(attrib1))
    # #
    # attrib2, latent = explain_single_packet_along_path(
    #     attack_data, target_idx, autoencoder, frac=0.1)
    #
    # draw_force_plot(attrib2.numpy(), feature_names,
    #                 "exp_figs/single2.png",  recons, encoded, name, latent)
    # print(tf.reduce_sum(attrib2))
    #
    # attrib3, latent = guided_ig(
    #      exp_attack_data, baseline, get_gradient(autoencoder), steps=2560, max_dist=1, fraction=1)
    #
    # draw_force_plot(attrib3.numpy(), feature_names,
    #                 "exp_figs/single3.png",  recons, encoded, name, latent)
    #
    # print(tf.reduce_sum(attrib3))

    all_attributions, all_positions, all_slopes = get_all_packet(
        baseline, attack_data, autoencoder, msteps=1024)
    draw_summary_plot(all_attributions.numpy(), attack_data,
                      feature_names, f"exp_figs/{ae_name}_summary_position.html", all_positions, bin_by_positions=True, bins=[10, 10])

    mean_attribution = np.mean(all_attributions, axis=0)
    mean_attribution /= np.linalg.norm(mean_attribution)
    print(np.array2string(mean_attribution, separator=', '))

    all_attributions, all_positions, all_slopes = get_all_packet(
        attack_data[:-1], attack_data[1:], autoencoder, msteps=30)

    # all_attributions = all_attributions / \
    #     tf.expand_dims(tf.norm(all_positions, axis=1), axis=1)
    draw_summary_plot(all_attributions.numpy(), attack_data[1:],
                      feature_names, f"exp_figs/{ae_name}_summary_slope.html", all_slopes, bin_by_positions=False, bins=36)