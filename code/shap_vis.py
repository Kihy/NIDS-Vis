from tqdm import tqdm
import shap
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper import *
from numpy.random import default_rng
import sklearn

def get_random_sample(seed, path, percentage=0.0012):
    rng = default_rng(seed)
    random_benign_sample=pd.read_csv(path, usecols=list(
        range(100)), skiprows=lambda x: x != 0 and rng.random() > percentage)
    print("number of background samples", random_benign_sample.shape)
    return random_benign_sample.to_numpy()

def get_boundary_sample(bd_path):
    boundary=np.genfromtxt(bd_path, delimiter=",")
    return boundary[0], boundary[1:-1], boundary[-1]

def get_shap_and_background(nids_model, feature, background):
    ex = shap.KernelExplainer(nids_model.predict, background)
    shap_values = ex.shap_values(feature)
    return shap_values, background.mean(axis=0)

def mean_inputation(k, shap_value, features, baseline):
    if k==0:
        return features
    
    imputed_features=np.copy(features)
    idx=np.argsort(np.abs(shap_value))[-k:]
    imputed_features[idx]=baseline[idx]
    return imputed_features

def calc_auc(shap_value, features, baseline):
    coord=[]
    for i in range(0,101,5):
        imputed_features_benign=mean_inputation(i, shap_value, features, baseline)
        score=nids_model.predict(imputed_features_benign)
        coord.append([i,score])
    coord=np.array(coord)
    difference=np.abs(coord[1:,1]-coord[:-1,1])
    normalised_diff=difference/np.max(difference)
    auc=sklearn.metrics.auc(coord[1:,0], normalised_diff)
    return auc, normalised_diff

def auc_test(nids_model, boundary_folder):

    random_benign_background=get_random_sample(42, "../../mtd_defence/datasets/uq/benign/Cam_1.csv", percentage=0.00015)
    boundary_aucs=[]
    benign_aucs=[]
    random_benign_ex = shap.KernelExplainer(nids_model.predict, random_benign_background)
    random_benign_baseline=random_benign_background.mean(axis=0)
    data_file_name=f"exp_csv/shap_auc_{boundary_folder.split('/')[-1]}.csv"
    with open(data_file_name,"w") as csv_file:
        for boundary in tqdm(os.listdir(boundary_folder)):        
            features, boundary_background, _ =get_boundary_sample(f"{boundary_folder}/{boundary}")
            shap_value, baseline=get_shap_and_background(nids_model, features, boundary_background)
            boundary_auc, boundary_diff=calc_auc(shap_value, features, baseline)
            boundary_aucs.append(boundary_auc)
            
            benign_shap_value = random_benign_ex.shap_values(features)
            benign_auc, benign_diff=calc_auc(benign_shap_value, features, random_benign_baseline)
            benign_aucs.append(benign_auc) 
            np.savetxt(csv_file, np.hstack([boundary_diff, boundary_auc, benign_diff, benign_auc])[np.newaxis,:], delimiter=",")
            csv_file.flush()
    print(np.mean(benign_aucs))
    print(np.mean(boundary_aucs))
    
        
def ablation_test(nids_model, boundary_file):
    random_benign_background=get_random_sample(42, "../../mtd_defence/datasets/uq/benign/Cam_1.csv", percentage=0.00015)
    # random_benign_background2=get_random_sample(23, "../../mtd_defence/datasets/uq/benign/Cam_1.csv")
    
    features, boundary_background, _ =get_boundary_sample(boundary_file)
    
    background_dict={"benign1":random_benign_background,
                    #  "benign2":random_benign_background2,
                     "boundary": boundary_background}
    
    fig, ax=plt.subplots(3,1, figsize=(5,15))
    feature_order=[]
    background_name=[]
    top_10_features=[]
    
    for name, background in tqdm(background_dict.items()):
        shap_value, baseline=get_shap_and_background(nids_model, features, background)
        coord=[]
        for i in range(0,101,5):
            imputed_features_benign=mean_inputation(i, shap_value, features, baseline)
            score=nids_model.predict(imputed_features_benign)
            coord.append([i,score])
        coord=np.array(coord)
        ax[0].plot(coord[:,0], coord[:,1], label=name)
        ordered_features=feature_names[np.argsort(np.abs(shap_value))[::-1]]
        feature_order.append(ordered_features)
        background_name.append(name)
        top_10_features.extend(ordered_features[:10])
        
        #plot the ablation difference
        difference=np.abs(coord[1:,1]-coord[:-1,1])
        normalised_diff=difference/np.max(difference)
        auc=sklearn.metrics.auc(coord[1:,0], normalised_diff)
        
        ax[2].plot(coord[1:,0], normalised_diff, label=f"{name} {auc[0]:.3f}")
        
    ax[0].set_title("Ablation Curve")
    ax[0].legend()
    
    feature_order=np.array(feature_order, dtype='object')
    top_10_features=set(top_10_features)
    
    for i in top_10_features:
        x_val=np.array(background_name, dtype='object')
        y_val=np.argmax(feature_order==i, axis=1)
        ax[1].plot(x_val, y_val)
        ax[1].text(-0.02, y_val[0]-0.1, f"{y_val[0]}: {i}", ha='right')
        ax[1].text(x_val.shape[0]-1+0.02, y_val[-1]-0.1, f"{y_val[-1]}: {i}")
    ax[1].set_title("Feature Ranking")
    ax[1].set_yscale("symlog",linthresh=10)
    ax[1].set_xlim(-1,x_val.shape[0])
    # remove spines
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_yticks([])
    ax[1].minorticks_off()
    
    ax[2].set_title("Attribution Difference")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig("exp_figs/shap_plots/ablation_test.png")
    

def plot_attributions(shap_values):
    plt.imshow(shap_values, interpolation=None)
    plt.xlabel("feature index")
    plt.ylabel("run number")
    plt.tight_layout()
    plt.savefig(f"exp_figs/shap_plots/random_background_summary.png")


def explain_adversarial(nids_model, boundary_file, feature_names, k=10, use_benign=False):
    adv, boundary_background, mal =get_boundary_sample(boundary_file)
    
    if use_benign:
        boundary_background=get_random_sample(42, "../../mtd_defence/datasets/uq/benign/Cam_1.csv", percentage=0.00015)
    
    ex = shap.KernelExplainer(nids_model.predict, boundary_background)
    adv_shap_values = ex.shap_values(adv)
    mal_shap_values=ex.shap_values(mal)
    
    sorted_adv_idx=np.argsort(np.abs(adv_shap_values))[::-1]
    top_adv_feat=feature_names[sorted_adv_idx]
    sorted_adv_shap=adv_shap_values[sorted_adv_idx]
    sorted_mal_idx=np.argsort(np.abs(mal_shap_values))[::-1]
    top_mal_feat=feature_names[sorted_mal_idx]
    sorted_mal_shap=mal_shap_values[sorted_mal_idx]
    top_10_features=np.union1d(top_adv_feat[:k], top_mal_feat[:k])
    feature_order=np.vstack([top_mal_feat,top_adv_feat])
    fig, ax=plt.subplots(1,1)
    for i in top_10_features:
        x_val=np.array(["mal","adv"], dtype='object')
        y_val=np.argmax(feature_order==i, axis=1)
        ax.plot(x_val, y_val)
        ax.text(-0.02, y_val[0]-0.1, f"{y_val[0]+1}: {i} {sorted_mal_shap[y_val[0]]:.3f}", ha='right')
        ax.text(x_val.shape[0]-1+0.02, y_val[-1]-0.1, f"{y_val[-1]+1}: {i} {sorted_adv_shap[y_val[1]]:.3f}")
    ax.set_title(f"Feature Ranking with {'Benign' if use_benign else 'DB'}")
    ax.set_yscale("symlog",linthresh=10)
    ax.set_xlim(-1,x_val.shape[0])
    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.minorticks_off()
    fig.savefig(f"exp_figs/shap_plots/{'ben' if use_benign else 'bd'}_feature_ranking.png")
    
    
    
if __name__=="__main__":

    nids_name="baseline_kitsune"
    atk_type="bt"
    idx=10
    t=0.28151878818499115
    run_name="ack_ben_none"
    dataset="uq"
    seed=42

    boundary_folder=f"../adversarial_data/Cam_1_{nids_name}_{atk_type}_{seed}_{run_name}_{t:.3f}"
    adv_boundary=f"../adversarial_data/Cam_1_{nids_name}_{atk_type}_{seed}_adv_ack_ack_ben_{t:.3f}/553.csv"
    # mal_path=f"../../mtd_defence/datasets/uq/malicious/Cam_1/Cam_1_ACK_Flooding.csv"
    
    nids_model= GenericADModel("baseline_kitsune",  **{
            "path": f"../../mtd_defence/models/{dataset}/kitsune/Cam_1.pkl",
            "func_name": "process", "threshold": t,
            "save_type": "pkl"})


    feature_names = np.array(["HT_MI_5_weight", "HT_MI_5_mean", "HT_MI_5_std", "HT_MI_3_weight", "HT_MI_3_mean", "HT_MI_3_std",
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
                    "HT_Hp_0.01_magnitude", "HT_Hp_0.01_covariance", "HT_Hp_0.01_pcc"], dtype='object')
    
    # mal_data=get_random_sample(42,mal_path)
    # auc_test(nids_model,boundary_folder)
    # ablation_test(nids_model, f"{boundary_folder}/0.csv")
    # explain_adversarial(nids_model, adv_boundary, feature_names, use_benign=True)
