import numpy as np
from helper import *
import pickle
import plotly.graph_objects as go
from KitNET.KitNET import KitNET
from tqdm import tqdm
import scipy
from scipy.signal import argrelextrema
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import argparse
from adv_based_vis import angle,take_step, ln_distance,read_adversarial_file, sample_n_from_csv
import itertools
import scienceplots
plt.style.use('science')

def train_nids(file, model_param, save_epoch=[1]):
    # train with 80% data, test with rest
    n_rows_epoch=file["total_rows"]

    FM_pkts=int(n_rows_epoch*0.2)
   
    
    if model_param["scaler"] is not None:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler=None
    
    nids = KitNET(100, 10, FM_pkts, 9999999, learning_rate=0.001, hidden_ratio=0.5, normalize=True)

    # nids = LocalOutlierFactor(
    #     n_neighbors=23, metric="euclidean", contamination=0.001, novelty=True)
    
    # nids=SOM("manhattan", "bubble", int(np.sqrt(5*np.sqrt(n_rows_epoch)))+1, 0.3, 0.5)
    
    traffic_ds = get_dataset(file["path"], model_param["batch_size"],
                             scaler=None, frac=1, read_with="tf", dtype="float32",
                             skip_header=True, shuffle=False)
    counter = 0

    for epoch in range(model_param["epochs"]):
        for data in tqdm(traffic_ds):
                
            data = data.numpy()
            if scaler:
                data = scaler.transform(data)
            
            # nids.fit(data)
            nids.train(data[0])
            
            counter += data.shape[0]
        print(epoch)
        if epoch+1 in save_epoch:
            with open(f"{model_param['path']}_{epoch+1}.pkl", "wb") as of:
                pickle.dump(nids, of)
                
            
            
    print(epoch, counter)
    return nids

def measure_feature_distance(dataset, network_atk, adv_atk, idx_offset,scaler):
    _,adv_feature,mal_feature=read_adversarial_file(dataset, network_atk, adv_atk, idx_offset)
    print(np.linalg.norm(scaler.transform(adv_feature)-scaler.transform(mal_feature),axis=1)[10])
    
def eval_nids(files, labels, nids_model, plot=False, full_data=True, scaler=None, file_handle=None):
    if plot:
        fig, axes=plt.subplots(len(files),1,figsize=(4,10), squeeze=False)
        iterable=zip(axes, files,labels)
    else:
        iterable=zip(files, labels)
    all_scores=[]
    all_labels=[]
    for i in iterable:
        if len(i)==2:
            file, label=i 
        else:
            ax, file, label=i
            ax=ax[0]
        if file["path"].endswith("npy"):
            array=np.load(file["path"])
            if scaler:
                array=scaler.inverse_transform(array)
            traffic_ds=np.array_split(array, array.shape[0]//1024)
            
        else:
            if full_data:
                frac=1
                total_rows=None
            else:
                frac=1
                total_rows=int(file["frac"]*file["total_rows"])
            traffic_ds = get_dataset(file["path"], 1024,
                                    scaler=None, frac=frac,total_rows=total_rows, read_with="tf", dtype="float32",
                                    skip_header=True, shuffle=False, drop_reminder=False)
            
        min_score=np.inf 

        
        scores = []
        
        for data in traffic_ds:
            if not isinstance(data, np.ndarray):
                data = data.numpy()
            score = nids_model.predict(data)
            if score.min()<min_score:
                min_score=score.min()
                
            scores.append(score)
        
        scores = np.hstack(scores)
        all_scores.append(scores)
        all_labels.append(np.full(scores.shape, label))
        
        # if file_handle:
        #     num_benign=np.sum(scores<nids_model.threshold)
        #     total=scores.size
        #     file_handle.write(f"{nids_model.abbrev},{file['abbrev']},{num_benign},{total},{num_benign/total}\n")
            
        if plot:
            sns.scatterplot(x=np.arange(scores.shape[0]), y=scores, alpha=0.1, s=2, ax=ax)
            ax.set_title(file['name'])
            ax.set_yscale("log")
            ax.axhline(nids_model.threshold, color="red")
            
        # print(f"min feature {repr(min_feature)} min score {min_score}")

    if file_handle:
        
        all_scores=np.hstack(all_scores)
        all_labels=np.hstack(all_labels)
        pred_labels=all_scores>nids_model.threshold 
        
        pr, re, t=sklearn.metrics.precision_recall_curve(all_labels, all_scores)
        opt_f1 = (2)*(pr*re)/(pr+re)
        optimal_idx=np.nanargmax(opt_f1)
        
        precision, recall, f1, support=sklearn.metrics.precision_recall_fscore_support(all_labels, pred_labels, average='binary')
        
        pr_fig, pr_ax = plt.subplots(figsize=(4, 4))
        display = sklearn.metrics.PrecisionRecallDisplay.from_predictions(all_labels, all_scores)
        f_scores = np.linspace(0.4, 0.9, num=5)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = pr_ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            
            pr_ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.8, y[49] + 0.02))
        pr_ax.set_xlim([0.0, 1.0])
        pr_ax.set_ylim([0.0, 1.05])    
        pr_ax.axhline(np.mean(all_labels))
        pr_ax.scatter(re[optimal_idx],pr[optimal_idx], marker='o', color='green', label=f'Best f {opt_f1[optimal_idx]:.3f}@{t[optimal_idx]:.4f}')
        pr_ax.scatter(recall, precision, marker='o', color='red', label=f'Actual f {f1:.3f}@{nids_model.threshold:.4f}')

        
        display.plot(ax=pr_ax, name=f"{nids_model.abbrev}")
        pr_fig.savefig(f"exp_figs/meta_plots/pr_curve/{nids_model.name}.png")
        plt.close(pr_fig)
        
        pr_auc = sklearn.metrics.average_precision_score(all_labels, all_scores)
        attr=nids_model.abbrev.split("-")
        model, epoch=attr
        file_handle.write(f"{model},{epoch},{pr_auc},{t[optimal_idx]},{opt_f1[optimal_idx]},{pr[optimal_idx]},{re[optimal_idx]},{nids_model.threshold},{f1}\n")

    # ax.axvline(split, color="red")
    if plot:
        fig.tight_layout()
        fig.savefig(f"exp_figs/meta_plots/anomaly_scores/{nids_model.name}_as.png")
        plt.close(fig)
    
    if len(files)==1:
        
        quantiles=[0.99,0.999,0.9999,1]
        thresholds=np.quantile(scores,quantiles)
        # fig=sns.displot(kind="hist", x=scores, log_scale=True, bins=20)
        
        shape, location, scale = scipy.stats.lognorm.fit(scores)
        ppf=scipy.stats.lognorm.ppf(0.99865,shape, location, scale)
        
        t_dict={str(k*100):v for k,v in zip(quantiles, thresholds)}
        t_dict["lognorm_3_std"]=ppf
        return t_dict
    else:
        return t[optimal_idx]

def adv_detect_exp(nids_model, file, feature_range, eps, p=2, scaler=None, write_to_file=True, full_data=True, 
                   near_threshold=False, exp_type="adv_detect", benign_dir=None):
    batch_size = 64

    if write_to_file:
        if exp_type=="train_split":
            csv_f=open(f"../data/{file['name']}_train.csv","w")
        else:
            csv_f = write_to_file
        verbose=False
    else:
        verbose=True
            
    if file["path"].endswith("csv"):
        if full_data:
            frac=1
            total_rows=None
        else:
            frac=file["frac"]
            total_rows=file["total_rows"]
        traffic_ds = get_dataset(file["path"], batch_size, total_rows=total_rows,
                                scaler=None, frac=frac, read_with="tf", dtype="float32",
                                skip_header=True, shuffle=False)
    elif file["path"].endswith("npy"):
        traffic_ds=np.load(file["path"])
        if scaler is not None:
            traffic_ds=scaler.inverse_transform(traffic_ds)
        traffic_ds=np.array_split(traffic_ds, traffic_ds.shape[0]//batch_size)

    counter = 0
    for data in tqdm(traffic_ds, position=0, desc=" data loop"):
        if not isinstance(data, np.ndarray):
            data=data.numpy()
            
        if not write_to_file:
           
            target_idx = np.array([545])
            
            if np.all(~((counter <= target_idx) & ((counter+data.shape[0]) > target_idx))):
                # print(counter)
                counter += data.shape[0]
                continue
            else:
                batch_idx = target_idx[np.where(
                    (target_idx-counter >= 0) & (target_idx-counter < batch_size))]
                batch_idx -= counter
                
            data = data[batch_idx]
            
        
        #filter by data
        idx = np.arange(counter, counter+data.shape[0])
        if near_threshold:
            scores=nids_model.predict(data)
            lower_idx=np.where((nids_model.threshold*0.9<scores) & (scores<nids_model.threshold*1.1))
            data=data[lower_idx]
            idx=idx[lower_idx]
        
        if data.size==0:
            counter += batch_size
            continue
        
        if exp_type=="robustness":
            
            result, largest_dir=robustness_measure(nids_model, data, feature_range, eps, p, benign_dir, verbose)
            
            if write_to_file:
                np.savetxt(csv_f, np.hstack([np.full((data.shape[0],1),file["abbrev"]), np.full((data.shape[0],1),eps), np.full((data.shape[0],1),p), idx[:,np.newaxis], result]),fmt="%s", delimiter=",")
                
            else:
                print(f"{file['name']}: {result}")
        elif exp_type=="train_split":
            if counter<=0.8*file["total_rows"]:
                np.savetxt(csv_f, data, delimiter=",")
            
        counter += batch_size

def dir_from_benign(benign_coord, data, feature_range, r):
    coordinates = np.tile(benign_coord,(data.shape[0],1,1))
    coordinates-=data[:,np.newaxis,:]
    coordinates/=feature_range 
    norm = np.linalg.norm(coordinates, p, axis=-1,keepdims=True)
    return coordinates/norm*r

def sample_from_ln_sphere(b, m, n, r, p):
    
    # Generate random coordinates from a standard normal distribution
    coordinates = np.random.normal(size=(b, m, n))
    
    # Calculate the Lp norm of the coordinates
    norm = np.linalg.norm(coordinates, p, axis=-1,keepdims=True)
    
    # Resample if the norm is zero
    while np.any(norm == 0):
        coordinates = np.random.normal(size=(b,m,n))
        norm = np.linalg.norm(coordinates, p, axis=-1,keepdims=True)
    
    # Scale the coordinates to lie on the surface of the sphere
    normalized_coordinates = coordinates / norm *r
    
    return normalized_coordinates

def robustness_measure(nids_model, data, feature_range, eps=0.3, p=2, benign_dir=None, verbose=False):
    n=200
    n_samples=50
    batch_size, d=data.shape
    
    if benign_dir is not None:
        sample=dir_from_benign(benign_dir, data, feature_range,eps)
    else:
        sample = sample_from_ln_sphere(batch_size, n, d, eps, p, benign_dir,data)
    steps = np.linspace(0, sample, n_samples) #shape=(n_samples, batch_size, n, d)
    steps=np.transpose(steps, (1,2,0,3))#shape=(batch_size, n, n_samples, d)
    
    features=data[:, np.newaxis, np.newaxis, :] + steps * \
        feature_range[np.newaxis,np.newaxis, np.newaxis, :]
    features=features.reshape((-1, d))
    res=nids_model.predict(features)
    res=res.reshape((batch_size, n, n_samples))
    labels=res>nids_model.threshold
    
    same_labels=(np.equal(labels,labels[:,:,0,np.newaxis]))
    adv_prob=np.mean(same_labels.reshape(same_labels.shape[0],-1), axis=1)
    
    last=np.argmax(same_labels[:,:,::-1],axis=2)

    adv_area=(np.sum(~same_labels, axis=2)-last)
    
    label_flips=np.diff(same_labels, axis=2)
    
    flips=np.sum(label_flips, axis=2)
    
    if verbose:
        idx=np.argpartition(flips, -2)[:,-2:]
        print(idx)
        print(nids_model.predict(data))
        print(repr(data))
        print(repr(sample[np.arange(data.shape[0]), idx]))
        print(repr(res[np.arange(data.shape[0]), idx]))
        print(same_labels[np.arange(data.shape[0]), idx])
        print(flips)
    largest_adv_dir=sample[np.arange(batch_size), np.argmax(flips, axis=1)]
    
    return np.hstack([res[:,0,0,np.newaxis],labels[:,0,0,np.newaxis], np.max(flips, axis=1,keepdims=True),np.mean(flips>1, axis=1,keepdims=True), np.mean(adv_area, axis=1, keepdims=True), adv_prob[:,np.newaxis]]), largest_adv_dir

    

    
    

def draw_scatter(csv_file, frac=1):
    df = pd.read_csv(csv_file, header=None, names=[
                     "idx", "as", "minima", "maxima"], skiprows=lambda x: np.random.random() > frac)

    # arr=np.genfromtxt(csv_file, delimiter=",")

    # The noise is in the range 0 to 0.5
    xnoise, ynoise = np.random.random(
        len(df))*0.8-0.4, np.random.random(len(df))*0.8-0.4
    df["minima_n"] = df["minima"]+xnoise
    df["maxima_n"] = df["maxima"]+ynoise

    fig = px.scatter(df, x="minima_n", y="maxima_n", hover_name="idx", hover_data=["as", "maxima", "minima"],
                     opacity=0.2,
                     color="as"
                     )
    # fig=px.bar(df.groupby(['maxima']).mean().reset_index(), x="maxima", y="as")

    fig.write_html("exp_figs/meta_plots/minima_test.html")

def shared_gradient_experiment(file_path, nids_model, feature_range):
    traffic_ds = get_dataset(file_path, 10,
                                scaler=None, frac=1, read_with="tf", dtype="float32",
                                skip_header=True, shuffle=False)
    for data in traffic_ds:
        data=data.numpy()
        gradient = gradient_estimate(
            nids_model.predict, data, delta_t=1e-5)/feature_range
        
        angles=angle(gradient, gradient)
        print(np.argmax(angles, axis=1))
        raise Exception
    
def gradient_descent(nids_model, x, feature_range):
    current_score=nids_model.predict(x)
    feature=np.copy(x)
    init_score=np.copy(current_score)

    prev_score=np.full(current_score.shape, np.inf)
    max_iter=1000

    # print("start",current_score)
    
    update_idx=np.full(current_score.shape,True)
    for i in tqdm(range(max_iter),desc=" grad loop", position=1, leave=False):
        prev_score[update_idx]=current_score[update_idx]
        
        gradient=gradient_estimate(nids_model.predict, feature[update_idx], delta_t=1e-5)/feature_range
        gradient /= np.linalg.norm(gradient, axis=1, keepdims=True)
        feature[update_idx]=take_step(feature[update_idx], np.array([0.05]), -gradient, feature_range)
        current_score[update_idx]=nids_model.predict(feature[update_idx])
        # print(current_score-prev_score)
    
        update_idx=(current_score-prev_score) < -1e-6
        if np.sum(update_idx)==0:
            break
        
    distance=ln_distance(x/feature_range, feature/feature_range, 2)
    return np.hstack([init_score[:,np.newaxis], distance])
        
def detect_adversarial(nids_model, x, feature_range, grad_range=1e-2, n_samples=201, print_scores=False, metrics=[]):
    gradient = gradient_estimate(
        nids_model.predict, x, delta_t=1e-5)/feature_range
    gradient /= np.linalg.norm(gradient, axis=1, keepdims=True)

    results=[]
    steps = np.linspace(-grad_range, grad_range, n_samples)

    features = x[:, np.newaxis, :]+gradient[:, np.newaxis, :] * \
        steps[np.newaxis, :, np.newaxis] * \
        feature_range[np.newaxis, np.newaxis, :]

    features = features.reshape([-1, 100])
    scores = nids_model.predict(features)
    scores = scores.reshape([x.shape[0], n_samples])
    mid_sample = n_samples//2

    score=scores[:, mid_sample]
    results.append(score)
    
    if "minima" in metrics:
        window_width = 2
        xp = np.pad(scores, ((0, 0), (1, 1)), mode='edge')

        cumsum_vec = np.cumsum(xp, axis=1)
        ma_vec = (cumsum_vec[:, window_width:] -
                cumsum_vec[:, :-window_width]) / window_width

        row_idx, _ = argrelextrema(ma_vec, np.less, axis=1)
        minima_count = np.zeros(scores.shape[0])
        for i in row_idx:
            minima_count[i] += 1
        results.append(minima_count)
    if "maxima" in metrics:
        row_idx, _ = argrelextrema(ma_vec, np.greater, axis=1)
        maxima_count = np.zeros(scores.shape[0])
        for i in row_idx:
            maxima_count[i] += 1
        results.append(maxima_count)
    
    if "pos_diff" in metrics:
        positive_diff=scores[:,-1]-score
        results.append(positive_diff)
        
    if "neg_diff" in metrics: 
        negative_diff=score-scores[:,0]
        results.append(negative_diff)
    
    if "curvature" in metrics:
        steps=np.tile(steps,(scores.shape[0],1))
        points=np.dstack([steps, scores])
        kappa=curvature(points[:,0,:],points[:,mid_sample,:],points[:,-1,:])
        results.append(kappa)
        
    if "convexity" in metrics:
        left=scores[:, mid_sample-1]-scores[:,mid_sample]
        right=scores[:,mid_sample+1]-scores[:,mid_sample]
        condlist=[(left>0) & (right>0),(left<0) & (right<0), (left>0)&(right<0), (left<0)&(right>0)]
        choicelist=[1,2,3,4]
        results.append(np.select(condlist, choicelist, default=0))
    
    if print_scores:
        print(repr(scores))
        print(results)
    return np.vstack(results).T

def det_minor(mat, i,j):
    minor=np.delete(np.delete(mat, i-1, -2),j-1,-1)
    
    return np.linalg.det(minor) 
    
def curvature(p1,p2,p3):
    batch_size=p1.shape[0]
    rows=[np.zeros((batch_size,1,4))]
    for p in [p1,p2,p3]:
        rows.append(np.dstack([np.sum(p**2, axis=1)[:,np.newaxis],p[:,0,np.newaxis],p[:,1,np.newaxis],np.ones((batch_size,1))]))
    matrix=np.hstack(rows)
    M11=det_minor(matrix, 1,1)
    kappa=np.where(np.abs(M11)<1e-6, 0,1/np.sqrt((0.5*det_minor(matrix, 1,2)/M11)**2+(-0.5*det_minor(matrix, 1,3)/M11)**2+det_minor(matrix, 1,4)/M11))
    
    return kappa*np.sign(M11)

def plot_adr(file):
    df = pd.read_csv(file)
    df["ADR"]=1-df["ADR"]
    fig=sns.catplot(data=df, x="file", y="ADR", hue="NIDS",kind="bar")
    fig.tight_layout()
    fig.savefig(f"exp_figs/adv_detect/adr.png")



if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='experiments with adversarial detection')
    parser.add_argument('--command', dest='command',
                        help='specify which command to run')
    args = parser.parse_args()
    
    scaler_path = f"../../mtd_defence/models/uq/autoencoder/Cam_1_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    feature_range = (scaler.data_max_ - scaler.data_min_)

    #"ACK-LM-0.1-10","ACK-LM-0.5-10","ACK-fgsm","ACK-deep-fool"
    files=["Cam_1", "ACK","SYN","UDP","PS","SD"]
    files=get_files(files)


    ae=[
        "denoising_autoencoder_sigmoid_2_D","autoencoder_relu_2_D" ,"autoencoder_sigmoid_2_D",
        "denoising_autoencoder_sigmoid_2_filtered_0.2","autoencoder_relu_2_filtered_0.2" ,"autoencoder_sigmoid_2_filtered_0.2",
        # "denoising_autoencoder_sigmoid_2_D","autoencoder_relu_2_D" ,"autoencoder_sigmoid_2_D",
        # "autoencoder_relu_2_DR" ,"autoencoder_sigmoid_2_DR","denoising_autoencoder_sigmoid_2_DR"
        "kitsune","autoencoder_relu_2" ,"autoencoder_sigmoid_2","denoising_autoencoder_sigmoid_2","autoencoder_sigmoid_25"
        ]
    # ae=["kitsune","autoencoder_relu_2","autoencoder_sigmoid_2","denoising_autoencoder_sigmoid_2"]
    # latent=[25,10,2 ] #10,2 "1","20","40","60","80","100"
    epochs=["1","20","40","60","80","100"]
    
    
    norms=[2]
    # eps_dict={('Kit', 1): 32.13307000394881, ('Kit', 20): 22.659977018882717, ('Kit', 40): 22.96192511784915, ('Kit', 60): 22.39158013680016, ('Kit', 80): 23.114019264607347, ('Kit', 100): 23.741728661085123, ('AE_R', 1): 37.02508372587951, ('AE_R', 20): 11.473390455890026, ('AE_R', 40): 9.49476189423244, ('AE_R', 60): 8.956378774133501, ('AE_R', 80): 7.844840656746276, ('AE_R', 100): 7.691986777580395, ('AE_25', 1): 45.043424811297065, ('AE_25', 20): 22.267088691892575, ('AE_25', 40): 20.436503457217825, ('AE_25', 60): 18.6647358235196, ('AE_25', 80): 17.624718714403706, ('AE_25', 100): 17.205741540048187, ('AE', 1): 44.94125811018481, ('AE', 20): 13.328929931502817, ('AE', 40): 8.826879032769602, ('AE', 60): 9.837247132923114, ('AE', 80): 9.305671198355066, ('AE', 100): 8.691413476262387, ('DAE', 1): 44.71580551996539, ('DAE', 20): 14.561238324145156, ('DAE', 40): 11.16767132014482, ('DAE', 60): 9.05845467839129, ('DAE', 80): 8.720464683730153, ('DAE', 100): 8.94059248630982}
    # eps_dict={('AE_RF0.1', 1): 44.52755979075273, ('AE_RF0.1', 20): 13.54439484633592, ('AE_RF0.1', 40): 11.946789522801534, ('AE_RF0.1', 60): 9.540309611673084, ('AE_RF0.1', 80): 8.132434085110843, ('AE_RF0.1', 100): 8.277069724304297, ('AEF0.1', 1): 45.03352581246152, ('AEF0.1', 20): 26.143281987954694, ('AEF0.1', 40): 12.305684805988607, ('AEF0.1', 60): 9.936083252059973, ('AEF0.1', 80): 8.882823484470508, ('AEF0.1', 100): 7.7713402125736515, ('DAEF0.1', 1): 36.93756785569306, ('DAEF0.1', 20): 27.003073038206225, ('DAEF0.1', 40): 14.692488347451487, ('DAEF0.1', 60): 11.963314360410092, ('DAEF0.1', 80): 10.69298255559674, ('DAEF0.1', 100): 8.548444346602437}
    eps_dict={('AE_RDR', 1): 28.658225772406027, ('AE_RDR', 20): 12.61660100201771, ('AE_RDR', 40): 7.859682825380768, ('AE_RDR', 60): 6.844487858814291, ('AE_RDR', 80): 5.485973219926727, ('AE_RDR', 100): 5.188774637235359, ('AEDR', 1): 43.77782658654176, ('AEDR', 20): 10.916199664924461, ('AEDR', 40): 7.639361239686056, ('AEDR', 60): 6.3371415284630706, ('AEDR', 80): 6.459030413003968, ('AEDR', 100): 6.437014758704065, ('DAEDR', 1): 42.753438012088836, ('DAEDR', 20): 9.10479279056355, ('DAEDR', 40): 7.023694428298578, ('DAEDR', 60): 6.1706193176972235, ('DAEDR', 80): 5.389549977345248, ('DAEDR', 100): 5.201740145561829}
    # eps_dict={('AE_RD', 1): 39.857077023063866, ('AE_RD', 20): 9.582735021421554, ('AE_RD', 40): 7.636253493658312, ('AE_RD', 60): 5.858337904265288, ('AE_RD', 80): 5.320559498343865, ('AE_RD', 100): 5.140652706797478, ('AED', 1): 41.695945718171096, ('AED', 20): 9.816203882142748, ('AED', 40): 7.214877274720111, ('AED', 60): 6.196857557309112, ('AED', 80): 5.530343504945706, ('AED', 100): 5.084077889204133, ('DAED', 1): 43.80351874470026, ('DAED', 20): 10.324876250009991, ('DAED', 40): 7.546845118926981, ('DAED', 60): 7.54316931068399, ('DAED', 80): 6.4312122031188625, ('DAED', 100): 5.766426095929945}
    if args.command=="adv_exp":
        idx, benign_directions=sample_n_from_csv(**files[0], n=200, seed=42)
        for name,  epoch in itertools.product(ae, epochs):
            
            model_name=f"{name}_{epoch}"
            
            nids_model=get_nids_model(model_name, threshold_key="99.9")
            eps=np.sqrt(eps_dict[(nids_model.abbrev.split("-")[0], int(epoch))])/2
            # eps=1.5
            # continue
            
            csv_f = open(f"exp_csv/adv_detect/{nids_model.name}_benign_dir.csv", "w")
            metrics=["file","eps","p","idx","score","label", "max_flips","flip_prob","adv_area","adv_prob"]
            csv_f.write(",".join(metrics))
            csv_f.write("\n")
            for file in files:
                for p in norms:
                    print(nids_model.name, file["abbrev"], eps, p)
                    adv_detect_exp(nids_model, file, feature_range, eps,p, scaler=scaler, write_to_file=csv_f,
                                   full_data=True, near_threshold=True, exp_type="robustness",benign_dir=benign_directions)
    
    elif args.command=="plot_res":
        # eval results
        df_auc=pd.read_csv("exp_csv/adv_detect/eval_results_loss.csv")
        df_auc.rename(columns={"nids":"NIDS"}, inplace=True)
        df_auc["filtered"]=df_auc["NIDS"].str.endswith("F0.2")
        df_auc["new_loss"]=df_auc["NIDS"].str.endswith("D")
        df_auc["NIDS"].replace({"F0.2":"", "D$":""},inplace=True, regex=True)
        
        # df_auc=df_auc.set_index(["NIDS","epoch"])
        
        # df_auc_melted=pd.melt(df_auc, value_vars=["opt_f1"],ignore_index=False)  
        # df_auc=df_auc.reset_index()
        print(df_auc.groupby(["epoch","NIDS","filtered","new_loss"]).agg("mean").to_csv())
        fig=sns.relplot(data=df_auc, x="epoch", y="opt_f1", hue="NIDS",kind="line", height=2, aspect=1.5,
                        facet_kws={"sharey":False})
        fig.set_xlabels("Epoch")
        fig.set_ylabels("Optimal F1")
        fig.tight_layout()
        fig.savefig("exp_figs/meta_plots/eval_summary_loss.pdf")
        
        
        results=[]
        for name, epoch in itertools.product(ae, epochs):
        # for name, epoch in zip(ae, best_epochs):
            model_name=f"{name}_{epoch}"
            nids_model=get_nids_model(model_name, threshold_key="99.9", load=False)
            result=pd.read_csv(f"exp_csv/adv_detect/{nids_model['name']}_benign_dir.csv")  
            result["nids"]=nids_model["abbrev"]
            results.append(result) 
        df=pd.concat(results)
        
        
        df[["NIDS","epoch"]]=df["nids"].str.rsplit("-", n=1, expand=True)
        df["filtered"]=df["NIDS"].str.endswith("F0.2")
        df["new_loss"]=df["NIDS"].str.endswith("DR")
        df["NIDS"].replace({"F0.2":"", "DR$":""},inplace=True, regex=True)

        
        df["robust"]=np.where(df["max_flips"]<=(1+df["label"]),1,0)
        
        df["benign"]=np.where(df["file"]=="Cam_1",True, False)
        df["epoch"]=df["epoch"].astype("int64")
        
        df=df.query("p==2")
        print(df.groupby(["epoch","NIDS","filtered","new_loss"]).agg("mean").to_csv())
        fig=sns.relplot(data=df, kind="line", x="epoch", y="robust",hue="NIDS", height=2, aspect=1.5, errorbar=None)
        fig.set_xlabels("Epoch")
        fig.set_ylabels("Robust \%")
        
        fig.tight_layout()
        fig.savefig("exp_figs/adv_detect/robust_perc.pdf")
        
        df["adv_prob"]=1-df["adv_prob"]
        fig=sns.relplot(data=df,kind="line", x="epoch",y="adv_prob",hue="nids",row="p", height=2.5, errorbar=None)
        fig.tight_layout()
        fig.savefig("exp_figs/adv_detect/adv_rob.png")
        
        
        # summary=df.groupby(["nids","epoch","p"]).agg("mean")
        # summary=summary.join(df_auc, how="left", on=["nids","epoch"])
        
        # best_idx=summary.groupby(['nids'])['opt_f1'].transform(max) == summary['opt_f1']
        # best_df=summary[best_idx]
        # best_df=best_df.reset_index()
        # fig=sns.catplot(kind="bar",data=best_df, x="nids",y="robust",hue="p",height=2.5)
        # fig.tight_layout()
        # fig.savefig("exp_figs/adv_detect/rob_best.png")
        
        
        # fig=sns.relplot(data=summary, x="PR-AUC",y="robust",hue="nids",row="p",height=2.5)
        # fig.tight_layout()
        # fig.savefig("exp_figs/adv_detect/pr-flip.png")
        
        # fig=sns.relplot(data=summary, x="opt_f1",y="robust",hue="nids",height=2.5,row="p")
        # fig.tight_layout()
        # fig.savefig("exp_figs/adv_detect/opt_f1-robust.png")
        
        
        # fig=sns.relplot(data=summary, x="real_f1",y="robust",hue="nids",height=2.5,row="p")
        # fig.tight_layout()
        # fig.savefig("exp_figs/adv_detect/real_f1-robust.png")
        
        
    elif args.command=="adv_dist":
        measure_feature_distance("uq","ACK_Flooding","autoencoder_0.1_10_3_False_pso0.5",854685, scaler=scaler)
        
    elif args.command=="train_nids":
        files=["Cam_1_train"]
        files=get_files(files)
        model_name="kitsune"
        save_epoch=[1,20,40,60,80,100]
        
        with open("configs/nids_models.json","r") as f:
            nids_db=json.load(f)
        # model_param_temp={
        #             "abbrev":"SOM-G",
		#             "path": "../models/SOM_gaussian",		            
        #             "func_name": "decision_function",
        #             "scaler": scaler_path,
        #             "flip_score": False,
        #             "perc":0.8,
        #             "epochs":100,
        #             "batch_size":1024,
        #             "save_type": "pkl"}
        
        model_param_temp={
                    "abbrev":"Kit",
		            "path": "../models/kitsune",		            
                    "func_name": "process",
                    "scaler": None,
                    "flip_score": False,
                    "perc":0.8,
                    "epochs":100,
                    "batch_size":1,
                    "save_type": "pkl"}
        
        train_nids(files[0], model_param_temp, save_epoch)
        
        for epoch in save_epoch:
            model_param=dict(model_param_temp)
            model_param["abbrev"]+=f"_{epoch}"
            model_param["path"]+=f"_{epoch}.pkl"
            nids_db[f"{model_name}_{epoch}"]=model_param 
            
        with open("configs/nids_models.json","w") as f:    
            json.dump(nids_db, f, indent=4)
        
        
    elif args.command=="find_threshold":
                
        with open("configs/nids_models.json","r") as f:
            nids_db=json.load(f)
        
        for name,  epoch in itertools.product(ae, epochs):
        
            model_name=f"{name}_{epoch}"
        
            nids_model=get_nids_model(model_name)
            model_param=nids_db[model_name]     
            
            files=["Cam_1_train"]
            files=get_files(files)        
            thresholds=eval_nids(files,[0], nids_model, plot=False, full_data=True, scaler=scaler)
            model_param["thresholds"]=thresholds
            
            nids_db.update({model_name:model_param})

            with open("configs/nids_models.json","w") as f:
                json.dump(nids_db, f, indent=4)
        
    elif args.command=="eval_nids":
        with open("configs/nids_models.json","r") as f:
            nids_db=json.load(f)
        files=["Cam_1","ACK","UDP","SYN","PS","SD"]
        files=get_files(files)
        with open("exp_csv/adv_detect/eval_results.csv","w") as results:
            results.write("nids,epoch,PR-AUC,opt_t,opt_f1,precision,recall,real_t,real_f1\n")
            for name, epoch in tqdm(itertools.product(ae, epochs)):
                
                nids_model=f"{name}_{epoch}"
                print(nids_model)
                nids_model=get_nids_model(nids_model,"99.9")
                opt_t=eval_nids(files,[0,1,1,1,1,1], nids_model, plot=True, full_data=False, scaler=scaler, file_handle=results)
                nids_db[nids_model.name]["thresholds"].update({"opt_t":float(opt_t)})
        with open("configs/nids_models.json","w") as f:
            json.dump(nids_db, f, indent=4)
        
                
    elif args.command=="plot_adr":
        plot_adr("exp_csv/adv_detect/ADR.csv")
        