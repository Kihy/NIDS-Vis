# from tqdm import tqdm
import shap
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng(0)

non_lin_ae_name = "FC_circular_max_1.01_sw_loss_distance_loss_model_example"
non_lin_ae_path = f"../models/FC_circular_max_1.01_sw_loss_distance_loss_model_example"


autoencoder = tf.keras.models.load_model(non_lin_ae_path)


scaler_path = "../../mtd_defence/models/uq/autoencoder/Cam_1_scaler.pkl"
benign_path = "../../mtd_defence/datasets/uq/benign/Cam_1.csv"
attack_path = "../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding.csv"

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)


background_data = pd.read_csv(benign_path, usecols=list(
    range(100)), skiprows=lambda x: x != 0 and rng.random() > 0.0001, dtype=np.float32)

attack_data = pd.read_csv(attack_path, usecols=list(
    range(100)), skiprows=lambda x: x != 0 and rng.random() > 0.0001, dtype=np.float32)


background_data = pd.DataFrame(scaler.transform(
    background_data), columns=background_data.columns)
attack_data = pd.DataFrame(scaler.transform(
    attack_data), columns=attack_data.columns)

mae = tf.keras.losses.MeanAbsoluteError(
    reduction=tf.keras.losses.Reduction.NONE)


# feature_names = ["HT_MI_5_weight", "HT_MI_5_mean,HT_MI_5_std", "HT_MI_3_weight", "HT_MI_3_mean", "HT_MI_3_std", "HT_MI_1_weight,HT_MI_1_mean", "HT_MI_1_std", "HT_MI_0.1_weight", "HT_MI_0.1_mean", "HT_MI_0.1_std,HT_MI_0.01_weight", "HT_MI_0.01_mean", "HT_MI_0.01_std", "HT_H_5_weight", "HT_H_5_mean", "HT_H_5_std", "HT_H_5_radius", "HT_H_5_magnitude", "HT_H_5_covariance", "HT_H_5_pcc", "HT_H_3_weight", "HT_H_3_mean", "HT_H_3_std", "HT_H_3_radius", "HT_H_3_magnitude", "HT_H_3_covariance", "HT_H_3_pcc", "HT_H_1_weight", "HT_H_1_mean", "HT_H_1_std", "HT_H_1_radius", "HT_H_1_magnitude", "HT_H_1_covariance", "HT_H_1_pcc", "HT_H_0.1_weight", "HT_H_0.1_mean", "HT_H_0.1_std", "HT_H_0.1_radius", "HT_H_0.1_magnitude", "HT_H_0.1_covariance", "HT_H_0.1_pcc", "HT_H_0.01_weight", "HT_H_0.01_mean", "HT_H_0.01_std", "HT_H_0.01_radius", "HT_H_0.01_magnitude", "HT_H_0.01_covariance", "HT_H_0.01_pcc", "HT_jit_5_weight",
#                  "HT_jit_5_mean", "HT_jit_5_std", "HT_jit_3_weight", "HT_jit_3_mean", "HT_jit_3_std", "HT_jit_1_weight", "HT_jit_1_mean", "HT_jit_1_std", "HT_jit_0.1_weight", "HT_jit_0.1_mean", "HT_jit_0.1_std", "HT_jit_0.01_weight", "HT_jit_0.01_mean", "HT_jit_0.01_std", "HT_Hp_5_weight", "HT_Hp_5_mean", "HT_Hp_5_std", "HT_Hp_5_radius", "HT_Hp_5_magnitude", "HT_Hp_5_covariance", "HT_Hp_5_pcc", "HT_Hp_3_weight", "HT_Hp_3_mean", "HT_Hp_3_std", "HT_Hp_3_radius", "HT_Hp_3_magnitude", "HT_Hp_3_covariance", "HT_Hp_3_pcc", "HT_Hp_1_weight", "HT_Hp_1_mean", "HT_Hp_1_std", "HT_Hp_1_radius", "HT_Hp_1_magnitude", "HT_Hp_1_covariance", "HT_Hp_1_pcc", "HT_Hp_0.1_weight", "HT_Hp_0.1_mean", "HT_Hp_0.1_std", "HT_Hp_0.1_radius", "HT_Hp_0.1_magnitude", "HT_Hp_0.1_covariance", "HT_Hp_0.1_pcc", "HT_Hp_0.01_weight", "HT_Hp_0.01_mean", "HT_Hp_0.01_std", "HT_Hp_0.01_radius", "HT_Hp_0.01_magnitude", "HT_Hp_0.01_covariance", "HT_Hp_0.01_pcc"]


def ae_wrapper(x):
    encoded, recon, decoded = autoencoder(x)
    return recon


explainer = shap.KernelExplainer(ae_wrapper, background_data)

idx = 0
shap_values = explainer.shap_values(attack_data)
shap.summary_plot(shap_values, attack_data, show=False)
plt.tight_layout()
plt.savefig('../plots/shap.png')
