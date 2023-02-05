from autoencoder import *
from measures import *

def random_rank_error(autoencoder,batch_size=128,feature_range=None):
    random_inputs=tf.random.uniform(shape=[batch_size,100],maxval=1)
    if feature_range is not None:
        random_inputs*=feature_range
    encoded, recon, _=autoencoder(random_inputs)
    dist=tf.linalg.norm(encoded, axis=1)
    recon_rank=tf.argsort(recon)
    dist_rank=tf.argsort(dist)


    return tf.losses.mae(recon_rank, dist_rank)

def rank_error(latent, recon):
    dist=tf.linalg.norm(latent, axis=1)
    recon_rank=tf.argsort(recon)
    dist_rank=tf.argsort(dist)


    return tf.losses.mae(recon_rank, dist_rank)

def eval_latent_space(autoencoder, scaler, dataset_path, batch_size=128):
    traffic_ds = get_dataset(dataset_path, batch_size,
                             scaler=None, frac=1, read_with="tf", dtype="float32",
                             seed=42, skip_header=True, shuffle=True)

    feature_range=scaler.data_max_ - scaler.data_min_

    metrics = {"lrmse": tf.keras.metrics.Mean(),
               "lmrrezx": tf.keras.metrics.Mean(),
               "lmrrexz": tf.keras.metrics.Mean(),
               "trust": tf.keras.metrics.Mean(),
               "conf": tf.keras.metrics.Mean(),
               "kl0.1": tf.keras.metrics.Mean(),
               "rre":tf.keras.metrics.Mean(),
               "re":tf.keras.metrics.Mean()
               }

    # Test autoencoder
    for data in tqdm(traffic_ds):

        latent, recon_error, decoded = autoencoder(data)
        results = MeasureCalculator(
            scaler.transform(data), latent, batch_size - 1)
        lmrrezx, lmrrexz = results.mrre(batch_size - 1)
        metrics["lrmse"].update_state(results.rmse())
        metrics["lmrrezx"].update_state(lmrrezx)
        metrics["lmrrexz"].update_state(lmrrexz)
        metrics["trust"].update_state(results.trustworthiness(25))
        metrics["conf"].update_state(results.continuity(25))
        metrics["kl0.1"].update_state(results.density_kl_global_01())
        metrics["rre"].update_state(random_rank_error(autoencoder, feature_range=feature_range, batch_size=batch_size))
        metrics["re"].update_state(rank_error(latent, recon_error))
    return metrics


if __name__ == '__main__':
    ae_names = [
        "min_max_mean_10.0_recon_loss_adam_denoising_2d_0.001",
        "min_max_mean_10.0_recon_loss_sw_loss_adam_denoising_2d_0.001",
        "min_max_mean_10.0_recon_loss_topological_loss_adam_denoising_2d_0.001",
        "min_max_mean_10.0_recon_loss_sliced_topo_loss_adam_denoising_2d_0.001",
        "min_max_mean_10.0_recon_loss_sw_loss_contractive_loss_adam_denoising_2d_0.001",
        "min_max_mean_10.0_recon_loss_contractive_loss_adam_denoising_2d_0.001",
        "min_max_mean_10.0_ranking_loss_sw_loss_dist_loss_adam_denoising_2d_0.001",
        "min_max_mean_10.0_ranking_loss_recon_loss_adam_denoising_2d_0.001",
        "min_max_mean_10.0_ranking_loss_dist_loss_adam_denoising_2d_0.001",
        "min_max_mean_10.0_ranking_loss_dist_loss2_adam_denoising_2d_0.001"
    ]
    output_file = "exp_csv/latent_metrics.csv"
    scaler_type = "min_max"
    scaler_path = f"../../mtd_defence/models/uq/autoencoder/Cam_1_{scaler_type}_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(output_file, "w") as f:
        f.write("AE, l-mse, l-mrrezx, l-mrrexz, trust, conf, kl0.1, rre, re\n")
        for ae_name in ae_names:
            ae_path = f"../models/{ae_name}"
            autoencoder = tf.keras.models.load_model(ae_path)

            metrics = eval_latent_space(
                autoencoder, scaler, "../../mtd_defence/datasets/uq/benign/Cam_1.csv", batch_size=512)
            f.write(
                f"{ae_name}, {metrics['lrmse'].result().numpy()}, {metrics['lmrrezx'].result().numpy()}, {metrics['lmrrexz'].result().numpy()}, {metrics['trust'].result().numpy()}, {metrics['conf'].result().numpy()}, {metrics['kl0.1'].result().numpy()}, {metrics['rre'].result().numpy()}, {metrics['re'].result().numpy()}\n")
