from tqdm import tqdm
import argparse
import shap
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper import *
from numpy.random import default_rng
import sklearn
import itertools
from adv_based_vis import sample_n_from_csv
from textwrap import wrap
import seaborn as sns

import scipy
import re

np.set_printoptions(threshold=np.inf)
rng=np.random.default_rng(42)

def get_random_sample(seed, path, percentage=0.0012):
    rng = default_rng(seed)
    random_benign_sample=pd.read_csv(path, usecols=list(
        range(100)), skiprows=lambda x: x != 0 and rng.random() > percentage)
    print("number of background samples", random_benign_sample.shape)
    return random_benign_sample.to_numpy()

def get_boundary_sample(bd_path):
    boundary=np.genfromtxt(bd_path, delimiter=",")
    return boundary[0], boundary[1:-3], boundary[-3:]

def get_shap_and_background(nids_model, feature, background):
    ex = shap.KernelExplainer(nids_model.predict, background)
    shap_values = ex.shap_values(feature)
    base_values= ex.expected_value
    return shap_values, base_values

def mean_inputation(k, shap_value, features, baseline, des=True):
    idx=np.argsort(np.abs(shap_value))
    if des:
        idx=idx[::-1]

    n_features=features.shape[0]
    mask=np.full((n_features+1, n_features), True)
    mask[np.triu_indices(n_features)]=False
    mask=mask[k]
    
    mask=mask[:,np.argsort(idx)]

    if baseline.ndim==2:
        mask=np.tile(mask, (baseline.shape[0],1,1))
        features=np.tile(features, (baseline.shape[0],k.shape[0],1))
        
        baseline=np.tile(baseline, (k.shape[0],1,1))
        baseline=baseline.transpose([1,0,2])
    else:
        features=np.tile(features, (k.shape[0], 1))
        baseline=np.tile(baseline, (k.shape[0], 1))
    

    features[mask]=baseline[mask]
    return features

def calc_auc(shap_value, features, baseline, k, des=True):
    imputed_features=mean_inputation(k, shap_value, features, baseline, des=des)
    imputed_features=imputed_features.reshape([-1,shap_value.shape[0]])
    score=nids_model.predict(imputed_features)
    score=score.reshape([baseline.shape[0],-1])
    difference=np.abs(score[:,1:]-score[:,:-1])
    
    auc=np.trapz(x=k[1:], y=difference)      
    return score, difference, auc

def calc_accuracy(nids_model, shap_values, feature, base_value, scaler, eps=0.01):
    if feature.ndim==1:
        feature=feature[np.newaxis,:]
    def explainability_model(x):
        perc_diff=(x-feature)/scaler.data_range_
        
        return np.einsum("ij,j->i",perc_diff+1, shap_values)+base_value


    n_noise=1000
    noise=rng.uniform(-eps, eps, (n_noise, 100))*scaler.data_range_ 
    neighbour_feature=np.clip(noise+feature, scaler.data_min_, scaler.data_max_)
    
    lin_diff=explainability_model(neighbour_feature)-nids_model.predict(neighbour_feature)
    return np.mean(np.abs(lin_diff)), nids_model.predict(feature)[0]
    
def auc_test(nids_model, boundary_folder, scaler, benign_background=False, start=None):
    run_name="/".join(boundary_folder.split('/')[2:])
    shap_back_file=open("exp_csv/explanability/shap_background.csv","a")
    shap_rand_file=open("exp_csv/explanability/shap_random.csv","a")
    shap_accuracy_file=open("exp_csv/explanability/shap_accuracy.csv","a")
   
    
    k=np.arange(0,101,5)
    neighbourhood=[1e-3,1e-2,0.1]
    
    if benign_background:
        background=get_random_sample(42, "../../mtd_defence/datasets/uq/benign/Cam_1.csv", percentage=1)
        background=shap.sample(background, 150)
        random_benign_ex = shap.KernelExplainer(nids_model.predict, background)
        run_name=f"Cam_1/{nids_model.name}_{nids_model.threshold:.3f}/42_{start}_sample_benign"
    if not os.path.exists(f"exp_csv/shap_values/{run_name}"):
        os.makedirs(f"exp_csv/shap_values/{run_name}")
    shap_values_file=open(f"exp_csv/shap_values/{run_name}/shap.csv","a")
    print(run_name)
        
    for idx, boundary in tqdm(enumerate(os.listdir(boundary_folder))):
        if benign_background:
            features, _, _ =get_boundary_sample(f"{boundary_folder}/{boundary}")
            shap_value=random_benign_ex.shap_values(features)
            base_value=random_benign_ex.expected_value
            
        else:
            features, background, _ =get_boundary_sample(f"{boundary_folder}/{boundary}")
            print(background.shape)
            if background.ndim==1 or background.size==0:
                continue
            if background.size>150:
                background=shap.sample(background, 150)
            shap_value, base_value=get_shap_and_background(nids_model, features, background)
        
        shap_values_file.write(f"{boundary[:-4]},")
        shap_values_file.write(np.array2string(shap_value,separator=',',max_line_width=np.inf)[1:-1]+"\n")
        shap_values_file.flush()
        
        score, _, auc =calc_auc(shap_value, features, background,k)
        shap_back_file.write(",".join([boundary.split(".")[0], run_name, np.array2string(score.mean(axis=0),separator=',',max_line_width=np.inf)[1:-1],np.array2string(score.std(axis=0),separator=',',max_line_width=np.inf)[1:-1], repr(auc.mean()),repr(auc.std())])) 
        shap_back_file.write("\n")
        shap_back_file.flush()
        
        for eps in neighbourhood:
            random_background=features + scaler.data_range_ * rng.uniform(-eps,eps, (150,100))
            random_background=np.clip(random_background, scaler.data_min_, scaler.data_max_)
            score, _, auc =calc_auc(shap_value, features, random_background,k, des=False)
            shap_rand_file.write(",".join([boundary.split(".")[0], run_name, str(eps), np.array2string(score.mean(axis=0),separator=',',max_line_width=np.inf)[1:-1],np.array2string(score.std(axis=0),separator=',',max_line_width=np.inf)[1:-1], repr(auc.mean()),repr(auc.std())])) 
            shap_rand_file.write("\n")
            shap_rand_file.flush()
            
            results=calc_accuracy(nids_model,shap_value,features,base_value, scaler, eps)
            shap_accuracy_file.write(",".join([boundary.split(".")[0], run_name, str(eps)]+list(map(str, results))) )
            shap_accuracy_file.write("\n")
            shap_accuracy_file.flush()
        
        
    shap_back_file.close()
    shap_rand_file.close()
    shap_accuracy_file.close()
    
    
def feature_order():
    feature_order=[]
    top_10_features=[]
    feature_order=np.array(feature_order, dtype='object')
    top_10_features=set(top_10_features)
    ordered_features=feature_names[np.argsort(np.abs(shap_value))[::-1]]
    feature_order.append(ordered_features)
    top_10_features.extend(ordered_features[:10])
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
        
def ablation_test(nids_model, boundary_file, plot_std=False):
    # random_benign_background2=get_random_sample(23, "../../mtd_defence/datasets/uq/benign/Cam_1.csv")

    features, boundary_background, _ =get_boundary_sample(boundary_file)
    
    background_dict={"boundary": boundary_background}
    
    for i in range(10):
        random_benign_background=get_random_sample(i, "../../mtd_defence/datasets/uq/benign/Cam_1.csv", percentage=0.00015*(i+1))
        background_dict[f"benign{i}"]=random_benign_background
        
    
    fig, ax=plt.subplots(2,1, figsize=(5,10))
    
    background_name=[]
    
    k=np.arange(0,101,5)
    
    for name, background in tqdm(background_dict.items()):
        shap_value=get_shap_and_background(nids_model, features, background)
        score, difference, auc =calc_auc(shap_value, features, background, k)
        
        ax[0].plot(k, score.mean(axis=0), label=name)
        if plot_std:
            ax[0].fill_between(k, score.mean(axis=0)-score.std(axis=0), score.mean(axis=0)+score.std(axis=0), alpha=0.2)
        background_name.append(name)
        
        #plot the ablation difference
        ax[1].plot(k[1:], difference.mean(axis=0), label=f"{name} {auc.mean():.3f}")
        
    ax[0].set_title("Ablation Curve")
    ax[0].legend()
    
    ax[1].set_title("Attribution Difference")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("exp_figs/shap_plots/ablation_test.png")
    

def plot_attributions(shap_values):
    plt.imshow(shap_values, interpolation=None)
    plt.xlabel("feature index")
    plt.ylabel("run number")
    plt.tight_layout()
    plt.savefig(f"exp_figs/shap_plots/random_background_summary.png")

def plot_average_ablation_difference(csv_file, eps=False):
    
    names=["idx","run_name"]
    if eps:
        names+=["eps"]
    
    data=pd.read_csv(csv_file, names=names+[f"average {i} imputed" for i in range(0,101,5)]+[f"std {i} imputed" for i in range(0,101,5)]+["mean auc","std auc"])
    data[["device","NIDS","Plane"]]=data["run_name"].str.split("/",expand=True)
    data=data[data["Plane"].str.startswith("42_Cam_1")]
    data=data.replace({"baseline_kitsune_0.283":"Kitsune","denoising_ae_0.086":"DAE","SOM_2.919":"SOM","vanilla_ae_0.066":"AE",
                       "42_Cam_1_Cam_1":"Interpolated","42_Cam_1_grad_attrib-ae":"Grad-AE","42_Cam_1_grad_attrib-pca":"Grad-PCA","42_Cam_1_random":"Random",
                       "42_Cam_1_sample_benign":"Sample"})
    n_runs=data["run_name"].nunique()
    n_rows=data["NIDS"].nunique()

    x_range=np.arange(0,101,5)
    
    if eps:
        for eps_name, eps_data in data.groupby("eps"):
            eps_data=eps_data.groupby(["NIDS","Plane"])
            ablation_curve_fig,_=plt.subplots(n_rows, n_runs//n_rows, figsize=(10,10), sharey=False)
            ablation_diff_fig,_=plt.subplots(n_rows, n_runs//n_rows, figsize=(10,10), sharey=False)
            
            for (name, df), ax_curve, ax_diff in zip(eps_data, ablation_curve_fig.axes, ablation_diff_fig.axes):
            
                anomaly_scores=df.iloc[:,3:24].to_numpy()
                # anomaly_scores=(anomaly_scores-np.min(anomaly_scores, axis=1, keepdims=True))/np.ptp(anomaly_scores, axis=1, keepdims=True)
                
                auc=np.trapz(anomaly_scores)

                ax_curve.plot(x_range, np.quantile(anomaly_scores,0.5, axis=0))
                # ax_curve.fill_between(x_range, np.quantile(anomaly_scores,0.1, axis=0), np.quantile(anomaly_scores,0.9, axis=0), alpha=0.2)
                ax_curve.set_title(f"{name}\n{auc.mean():.2f}")
                
                
                diff=np.abs(np.diff(anomaly_scores,axis=1))
                mean_diff=diff.mean(axis=0)
                auc=np.trapz(diff)
                
                ax_diff.plot(x_range[1:], mean_diff)
                # ax_diff.fill_between(x_range[1:], mean_diff-ci, mean_diff+ci, alpha=0.2)
                ax_diff.set_title(f"{name}\n{auc.mean():.2f}")
                
            ablation_curve_fig.tight_layout()
            file_name=csv_file.split("/")[-1][:-4]
            ablation_curve_fig.savefig(f"exp_figs/shap_plots/{file_name}_{eps_name}_curve.png")
            ablation_diff_fig.tight_layout()
            ablation_diff_fig.savefig(f"exp_figs/shap_plots/{file_name}_{eps_name}_diff.png")
        # handles, labels = ax_curve.get_legend_handles_labels()
        # ablation_curve_fig.legend(handles, labels, loc='upper center')
    else:
        ablation_curve_fig,_=plt.subplots(n_rows, n_runs//n_rows, figsize=(10,10), sharey=False)
        ablation_diff_fig,_=plt.subplots(n_rows, n_runs//n_rows, figsize=(10,10), sharey=False)
        for (name, run_data), ax_curve, ax_diff in zip(data.groupby(["NIDS","Plane"]), ablation_curve_fig.axes, ablation_diff_fig.axes):
            
            anomaly_scores=run_data.iloc[:,2:23].to_numpy()
            
            # anomaly_scores=(anomaly_scores-np.min(anomaly_scores, axis=1, keepdims=True))/np.ptp(anomaly_scores, axis=1, keepdims=True)
        
            mean_as=anomaly_scores.mean(axis=0)
            
            auc=np.trapz(anomaly_scores)
            
            ax_curve.plot(x_range, mean_as)
            # ax_curve.fill_between(x_range, mean_as-std_as, mean_as+std_as, alpha=0.2)
            ax_curve.set_title(f"{name}\n{auc.mean():.2f}")
            
            
            diff=np.abs(np.diff(anomaly_scores,axis=1))
            mean_diff=diff.mean(axis=0)
            std_diff=diff.std(axis=0)
            auc=np.trapz(diff)
            
            ax_curve.plot(x_range, np.quantile(anomaly_scores,0.5, axis=0))
            ax_curve.fill_between(x_range, np.quantile(anomaly_scores,0.1, axis=0), np.quantile(anomaly_scores,0.9, axis=0), alpha=0.2)
            ax_diff.set_title(f"{name}\n{auc.mean():.2f}")
            
            
            ablation_curve_fig.tight_layout()
            file_name=csv_file.split("/")[-1][:-4]
            ablation_curve_fig.savefig(f"exp_figs/shap_plots/{file_name}_curve.png")
            ablation_diff_fig.tight_layout()
            ablation_diff_fig.savefig(f"exp_figs/shap_plots/{file_name}_diff.png")

def plot_accuracy(csv_file):
    data=pd.read_csv(csv_file)
    data[["device","NIDS","Plane"]]=data["run_name"].str.split("/",expand=True)
    data=data[data["Plane"].str.startswith("42_Cam_1")]
    data=data.replace({"baseline_kitsune_0.283":"Kit","denoising_ae_0.086":"DAE","SOM_2.919":"SOM","vanilla_ae_0.066":"AE",
                       "42_Cam_1_Cam_1":"Interpolated","42_Cam_1_grad_attrib-ae":"IG-AE","42_Cam_1_grad_attrib-pca":"IG-PCA","42_Cam_1_random":"Random",
                       "42_Cam_1_sample_benign":"Sample"})

    data["threshold"]=np.where(data["NIDS"]=="Kitsune", 0.283, 2.919)
    data["diff (%)"]=data["accuracy"]/data["score"]
    
    
    fig=sns.catplot(data=data, x="diff (%)",col="eps",y="Plane",row="NIDS", sharex="none", kind="bar", height=2.2,aspect=1)
    # line_position = [0.05, 0.1, 0.15, 0.2]
    # for ax, pos in zip(fig.axes.flat, line_position):
    #     ax.axvline(x=pos, color='r', linestyle=':')
        
    file_name=csv_file.split("/")[-1][:-4]
    fig.savefig(f"exp_figs/shap_plots/{file_name}_accuracy.png")
    
    fig2=sns.catplot(data=data, x="recon_error",col="eps",hue="Plane",y="NIDS", sharex="none", kind="bar", height=3,aspect=1)
    fig2.savefig(f"exp_figs/shap_plots/{file_name}_recon.png")
    
    fig3=sns.catplot(data=data, x="recon_error_normalised",col="eps",hue="Plane",y="NIDS", sharex="none", kind="bar", height=3,aspect=1)
    fig3.savefig(f"exp_figs/shap_plots/{file_name}_recon_norm.png")
    
    fig4=sns.relplot(data=data, x="eps",col="NIDS",hue="Plane",y="diff (%)",  kind="line", height=3,aspect=1, facet_kws={"sharey":"none"})
    fig4.savefig(f"exp_figs/shap_plots/{file_name}_line.png")
    
    

def explain_adversarial(nids_model, boundary_folder, feature_names, k=10):
    all_shap_values=[]
    all_feature_values=[]
        
    for boundary_file in tqdm(os.listdir(boundary_folder)):
        target, boundary_background, _ =get_boundary_sample(f"{boundary_folder}/{boundary_file}")
        all_shap_values.append(get_shap_and_background(nids_model, target, boundary_background))
        all_feature_values.append(target)
        
        
    all_shap_values=np.vstack(all_shap_values)
    all_feature_values=np.vstack(all_feature_values)
    all_score=nids_model.predict(all_feature_values)
    
    top_feature_idx=np.argsort(np.sum(np.abs(all_shap_values), axis=0))[::-1]
    
    fig, _=plt.subplots(int(np.sqrt(k)),int(np.sqrt(k)), figsize=(11,10))
    
    for i, ax in enumerate(fig.axes):
        im=ax.scatter(all_shap_values[:,top_feature_idx[i]], all_feature_values[:,top_feature_idx[i]], c=all_score, alpha=0.5)
        ax.set_title("\n".join(wrap(f"{i+1}: {feature_names[top_feature_idx[i]]}", 30)))

        ax.set_ylabel("feature value")
        ax.set_xlabel("shap value")
        
        fig.colorbar(im, ax=ax)
    
    
    fig.tight_layout()
    fig.savefig(f"exp_figs/shap_plots/explain_{boundary_folder.split('_0.282_')[-1]}.png")

def consistency(file_regex, out_file_name):
    shap_values=[]
    for file in os.listdir("exp_csv/shap_values"):
        if re.match(file_regex, file):
            print(f"exp_csv/shap_values/{file}")
            data=np.genfromtxt(f"exp_csv/shap_values/{file}",delimiter=",")
            if data.shape[0]>1000:
                data=np.split(data, data.shape[0]//1000)
                shap_values.extend(data)
            else:
                shap_values.append(data)
    shap_values=np.dstack(shap_values)
    
    distances=[np.mean(scipy.spatial.distance.pdist(shap_values[i].T,'cityblock')) for i in range(1000)]
    # print(distances.mean(), distances.std())
    sns.boxplot(x=distances, showfliers=True, showmeans=False)
    plt.tight_layout()
    plt.savefig(f"exp_figs/shap_plots/{out_file_name}_consistency.png")
    
def mean_feature(file):
    data=pd.read_csv(file, usecols=range(100))
    data.hist(figsize=(20,20),bins=100)
    plt.tight_layout()
    plt.savefig("exp_figs/meta_plots/dist.png")
    print(repr(data.mean(axis=0, numeric_only=True).to_numpy()))
    print(repr(data.median(axis=0,numeric_only=True).to_numpy()))
    
    
    
if __name__=="__main__":
    parser=argparse.ArgumentParser(description='experiments with shap')
    parser.add_argument('--command', dest='command',
                        help='specify which command to run')
    args = parser.parse_args()
    
    
    # nids_name="ae"
    idx=10
    t=0.282
    run_name="ben_none_none"
    dataset="uq"
    seed=42
    init_search="linear"
    tangent_guide="anchor"
    device_name="Cam_1"
    
    
    # adv_boundary=f"../adversarial_data/Cam_1_{nids_name}_{seed}_adv_ack_ack_ben_{t:.3f}/553.csv"
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
    
    time_windows={"5":"0.1s", "3":"0.5s","1":"1.5s","0.1":"10s","0.01":"1m"}
    relations={"MI":"ps from srcMAC&srcIP","jit":"iat between srcIP to dstIP","H":"between ps from srcIP to dstIP","Hp":"ps between srcIP&port to dstIP&port"}
    stats={"weight":"freq of","mean": "mean","std":"std of", "magnitude":"bi-directional mean of", "radius":"bi-directional var of", "covariance": "relationship between","pcc":"scaled relationship between"}
    
    modified_feature_names=[]
    for i in feature_names:
        components=i.split("_")
        new_name=f"{stats[components[-1]]} {relations[components[-3]]} in {time_windows[components[-2]]}"
        modified_feature_names.append(new_name)
    
    scaler_path = f"../../mtd_defence/models/uq/autoencoder/Cam_1_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    if args.command=="run_auc_exp":
        
        start=["Cam_1","ACK-LM-0.5-10","ACK-LM-0.1-10"]
        nids_models=[get_nids_model(i) for i in ["vanilla_ae", "denoising_ae","baseline_kitsune","SOM"]]
        planes=["Cam_1","grad_attrib-pca","grad_attrib-ae","random"]

        for nids_model in nids_models:
            boundary_folders=[]
            for s in start:
                for p in planes:
                    if s=="ACK-LM-0.5-10" and p=="Cam_1":
                        continue
                    boundary_folder=f"../adversarial_data/{device_name}/{nids_model.name}_{nids_model.threshold:.3f}/{seed}_{s}_{p}"
            
                    # auc_test(nids_model,boundary_folder,scaler)
                auc_test(nids_model, boundary_folder, scaler,benign_background=True, start=s)
        
    elif args.command=="plot_result":
        plot_average_ablation_difference("exp_csv/explanability/shap_background.csv")
        plot_average_ablation_difference("exp_csv/explanability/shap_random.csv", eps=True)
        plot_accuracy("exp_csv/explanability/shap_accuracy.csv")
    elif args.command=="consistency":
        consistency(f"{device_name}_{nids_model.name}_42_{nids_model.threshold:.3f}_{device_name}_.*_grad_attrib-pca","v2=F-pca")
    elif args.command=="mean_data":
        mean_feature("../../mtd_defence/datasets/uq/benign/Cam_1.csv")
    # explain_adversarial(nids_model, f"../adversarial_data/{device_name}_{nids_name}_{seed}_{t:.3f}_ben_gradient_grad_attrib", modified_feature_names)
    # explain_adversarial(nids_model, f"../adversarial_data/{device_name}_{nids_name}_{seed}_{t:.3f}_mod_adv_gradient_grad_attrib", modified_feature_names)
    # explain_adversarial(nids_model, f"../adversarial_data/{device_name}_{nids_name}_{seed}_{t:.3f}_adv_gradient_grad_attrib", modified_feature_names)
    