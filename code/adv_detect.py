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
from adv_based_vis import angle,take_step, ln_distance,read_adversarial_file, sample_n_from_csv, reservoir_sample
import itertools
import scienceplots
plt.style.use('science')
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
matplotlib.rcParams.update({'font.size': 10})
def train_nids(file, model_param, save_epoch=[1]):
    # train with 80% data, test with rest
    n_rows_epoch=file["total_rows"]

    FM_pkts=int(n_rows_epoch*0.2)
   
    
    if model_param["scaler"] is not None:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler=None
    
    nids = KitNET(model_param["input_dim"], 10, FM_pkts, 9999999, learning_rate=0.001, hidden_ratio=0.5, normalize=True)

    # nids = LocalOutlierFactor(
    #     n_neighbors=23, metric="euclidean", contamination=0.001, novelty=True)
    
    # nids=SOM("manhattan", "bubble", int(np.sqrt(5*np.sqrt(n_rows_epoch)))+1, 0.3, 0.5)
    
    traffic_ds = get_dataset(file["path"], model_param["batch_size"],ndim=model_param["input_dim"],
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
    
def eval_nids(dataset, files, labels, nids_model, plot=False, full_data=True, scaler=None, file_handle=None):
    if plot:
        fig, axes=plt.subplots(len(files),1,figsize=(4,10), squeeze=False)
        iterable=zip(axes, files,labels)
    else:
        iterable=zip(files, labels)
    all_scores=[]
    all_labels=[]
    for i,data in enumerate(iterable):
        if len(data)==2:
            file, label=data 
        else:
            ax, file, label=data
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
            traffic_ds = get_dataset(file["path"], 1024, ndim=46,
                                    scaler=None, frac=frac,total_rows=total_rows, read_with="tf", dtype="float32",
                                    skip_header=True, shuffle=False, drop_reminder=False)
            
        scores = []
        
        for data in traffic_ds:
            if not isinstance(data, np.ndarray):
                data = data.numpy()
            
            score = nids_model.predict(data)
           
                
            scores.append(score)
        
        scores = np.hstack(scores)
        all_scores.append(scores)
        all_labels.append(np.full(scores.shape, label))
        
        if i==0:
            quantiles=[0.99,0.999,0.9999,1]
            thresholds=np.quantile(scores,quantiles)
            # fig=sns.displot(kind="hist", x=scores, log_scale=True, bins=20)
            
            shape, location, scale = scipy.stats.lognorm.fit(scores)
            ppf=scipy.stats.lognorm.ppf(0.99865,shape, location, scale)
            
            t_dict={str(k*100):v for k,v in zip(quantiles, thresholds)}
            t_dict["lognorm_3_std"]=ppf
        
        # if file_handle:
        #     num_benign=np.sum(scores<nids_model.threshold)
        #     total=scores.size
        #     file_handle.write(f"{nids_model.abbrev},{file['abbrev']},{num_benign},{total},{num_benign/total}\n")
            
        if plot:
            sns.scatterplot(x=np.arange(scores.shape[0]), y=scores, alpha=0.1, s=2, ax=ax)
            ax.set_title(file['name'])
            ax.set_yscale("log")
            ax.axhline(t_dict["99.9"], color="red")
            
        # print(f"min feature {repr(min_feature)} min score {min_score}")

    if file_handle:
        
        all_scores=np.hstack(all_scores)
        all_labels=np.hstack(all_labels)
        pred_labels=all_scores>t_dict["99.9"]
        
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
        pr_ax.scatter(recall, precision, marker='o', color='red', label=f'Actual f {f1:.3f}@{t_dict["99.9"]:.4f}')
        pr_ax.plot(re, pr, label="pr_curve")
        

        
        display.plot(ax=pr_ax, name=f"{nids_model.abbrev}")
        pr_fig.savefig(f"exp_figs/meta_plots/pr_curve/{dataset}/{nids_model.name}.png")
        plt.close(pr_fig)
        
        t_dict["opt_t"]=float(t[optimal_idx])
        pr_auc = sklearn.metrics.average_precision_score(all_labels, all_scores)
        attr=nids_model.abbrev.split("-")
        model, epoch=attr
        file_handle.write(f"{model},{epoch},{pr_auc},{t[optimal_idx]},{opt_f1[optimal_idx]},{pr[optimal_idx]},{re[optimal_idx]},{t_dict['99.9']},{f1}\n")

    # ax.axvline(split, color="red")
    if plot:
        fig.tight_layout()
        fig.savefig(f"exp_figs/meta_plots/anomaly_scores/{dataset}/{nids_model.name}_as.png")
        plt.close(fig)
    
    plt.close('all')
    
    return t_dict
    

def adv_detect_exp(nids_model, idx, file_data, file_name, feature_range, eps, p=2, write_to_file=False,
                   benign_dir=None):
    batch_size = 32
    
    n_batches=file_data.shape[0]//batch_size+1
    
    if write_to_file:
        verbose=False
    else:
        verbose=True


    for i, data, file in tqdm(zip(np.array_split(idx, n_batches),np.array_split(file_data, n_batches),np.array_split(file_name, n_batches)), position=0, desc=" data loop"):
        
        result=robustness_measure(nids_model, data, feature_range, eps, p, benign_dir, verbose)
        
        if write_to_file:
            np.savetxt(csv_f, np.hstack([file[:,np.newaxis], np.full((data.shape[0],1),eps), np.full((data.shape[0],1),p), i[:,np.newaxis], result]),fmt="%s", delimiter=",")
        else:
            np.savetxt("exp_csv/performance/test.csv",result, delimiter=",")
        
        
def dir_from_benign(benign_coord, data, feature_range, r):
    coordinates = np.tile(benign_coord,(data.shape[0],1,1))
    coordinates-=data[:,np.newaxis,:]
    coordinates/=feature_range 
    norm = np.linalg.norm(coordinates, p, axis=-1,keepdims=True)
    return coordinates/norm*r

def orthonormal_basis(b,n,r,p):
    
    # Generate a random orthogonal matrix
    rand_ortho_matrix = np.random.randn(b, n, n)
    q, _ = np.linalg.qr(rand_ortho_matrix)
    norm = np.linalg.norm(q, p, axis=-1,keepdims=True)
    normalized_coordinates = q / norm *r
    return normalized_coordinates

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
    n=1000
    n_samples=20
    batch_size, d=data.shape
    
    if benign_dir is not None:
        sample=dir_from_benign(benign_dir, data, feature_range,eps)
    else:
        sample = sample_from_ln_sphere(batch_size, n, d, eps, p)
        # sample=orthonormal_basis(batch_size, d, eps, p)
        # n=d
    steps = np.linspace(0, sample, n_samples) #shape=(n_samples, batch_size, n, d)
    steps=np.transpose(steps, (1,2,0,3))#shape=(batch_size, n, n_samples, d)
    
    features=data[:, np.newaxis, np.newaxis, :] + steps * \
        feature_range[np.newaxis,np.newaxis, np.newaxis, :]
    res=nids_model.predict(features.reshape((-1, d)))
    res=res.reshape((batch_size, n, n_samples))
    #find turning points 
    
    dx = np.diff(res, axis=-1)
    mini=(dx[:, :, 1:] * dx[:,:,:-1] < 0) &(dx[:, :, 1:]<0)
    maxi=(dx[:, :, 1:] * dx[:,:,:-1] < 0) &(dx[:, :, 1:]>0)

    maxima_idx=np.where(mini)
    minima_idx=np.where(maxi)
    

    max_mask=np.full(res.shape, True)
    max_mask[maxima_idx[0], maxima_idx[1], maxima_idx[2]+1]=False 
    max_mask[res<(nids_model.threshold/2)]=True
    masked_max=np.ma.masked_array(res, mask=max_mask)

    min_mask=np.full(res.shape, True)
    min_mask[minima_idx[0], minima_idx[1], minima_idx[2]+1]=False
    min_mask[res<(nids_model.threshold/2)]=True
    masked_min=np.ma.masked_array(res, mask=min_mask)

    masked_min=np.ma.masked_where(masked_min == np.min(masked_min, axis=-1, keepdims=True), masked_min)


    maxima=np.count_nonzero(masked_max, axis=-1)
    minima=np.count_nonzero(masked_min, axis=-1)
    
    
    turning_points=maxima.filled(0)+minima.filled(0)
    
    labels=res>nids_model.threshold
    
    same_labels=(np.equal(labels,labels[:,:,0,np.newaxis]))
    adv_prob=np.mean(same_labels.reshape(same_labels.shape[0],-1), axis=1)
    
    last=np.argmax(same_labels[:,:,::-1],axis=2)

    adv_area=(np.sum(~same_labels, axis=2)-last)
    
    label_flips=np.diff(same_labels, axis=2)
    
    flips=np.sum(label_flips, axis=2)
    
    if verbose:
        metric=flips
        
        print("score",  nids_model.predict(data))
        # print("data",repr(data))
        # print("vul. dir", repr(sample[np.arange(data.shape[0]), idx]))
        
        print("number of flips",flips)
        print("number of turning points",turning_points)
        
        # print("dx along vul. dir", repr(dx[np.arange(data.shape[0]), idx]))
        print(eps)
        
        idx=np.argpartition(metric, -2)[:,-2:]
        print(idx)

        #
        all_idx=np.where(metric>1+labels[:,0,0])
        idx=np.random.choice(all_idx[1], 2, replace=False)[np.newaxis,:]
        print(all_idx)
        print(idx)
        
        
        print("turning points along vul. dir", repr(turning_points[np.arange(data.shape[0]), idx]))
        print("score along vul. dir", repr(res[np.arange(data.shape[0]), idx]))
        print("same label along vul. dir",same_labels[np.arange(data.shape[0]), idx])
        print("l2 dist", np.linalg.norm(sample[np.arange(data.shape[0]), idx], axis=-1))
        
        
        # sample=sample[np.arange(data.shape[0]), idx][0]
        sample=sample[all_idx]
        sample/=np.linalg.norm(sample, axis=-1, keepdims=True)
        # return np.vstack([data, sample, features[np.arange(data.shape[0]), idx,:,:].reshape(-1,d)])
        return np.vstack([data, sample])
    
    return np.hstack([res[:,0,0,np.newaxis],labels[:,0,0,np.newaxis], np.max(flips, axis=1,keepdims=True),np.mean(flips>1, axis=1,keepdims=True), np.mean(adv_area, axis=1, keepdims=True), adv_prob[:,np.newaxis], np.mean(turning_points, axis=1, keepdims=True),np.max(turning_points, axis=1, keepdims=True)])

    

    
    

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
    
    dataset="Smartphone_1"
    scaler_path = f"../../mtd_defence/models/uq/autoencoder/{dataset}_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    if not os.path.exists(f"exp_csv/adv_detect/{dataset}"):
        os.mkdir(f"exp_csv/adv_detect/{dataset}")
    feature_range = (scaler.data_max_ - scaler.data_min_)
    
    #"ACK-LM-0.1-10","ACK-LM-0.5-10","ACK-fgsm","ACK-deep-fool"  "ACK","SYN","UDP","PS","SD"
    files=[f"{dataset}",f"{dataset}_ACK",f"{dataset}_UDP",f"{dataset}_SYN",f"{dataset}_PS",f"{dataset}_SD"]
    # files=[f"{dataset}",f"{dataset}_ATK"]
    files=get_files(files)


    ae=["denoising_autoencoder_sigmoid_2_filtered_0.2","autoencoder_relu_2_filtered_0.2" ,"autoencoder_sigmoid_2_filtered_0.2", "kitsune_filtered_0.2",
        "autoencoder_relu_2" ,"autoencoder_sigmoid_2","denoising_autoencoder_sigmoid_2",
        "autoencoder_sigmoid_25","kitsune",
        "denoising_autoencoder_sigmoid_2_D","autoencoder_relu_2_D" ,"autoencoder_sigmoid_2_D",
        # "autoencoder_sigmoid_2","denoising_autoencoder_sigmoid_2","autoencoder_sigmoid_25"
        # "kitsune"
        ]

    # ae=["kitsune","autoencoder_relu_2","autoencoder_sigmoid_2","denoising_autoencoder_sigmoid_2"]
    # latent=[25,10,2 ] #10,2 "1","20","40","60","80","100"
    epochs=["1","20","40","60","80","100"]
    norms=[1,2,np.inf]
    
    if args.command=="adv_exp":
        
        nt=[False]
        
        test_mode=False
        # file_idx=0
        # target_idx=[133445]
        # file_names=[files[file_idx]["abbrev"]]
        # start_samples = pd.read_csv(files[file_idx]["path"], usecols=list(range(100)), skiprows = lambda x: x-1 not in target_idx,header=None).to_numpy()
        
        for near_threshold in nt:
            if not test_mode and not near_threshold:
                target_idx, start_samples, file_names=reservoir_sample(files, n=2000, near_boundary=near_threshold)
                
            for name,  epoch in itertools.product(ae, epochs):
                # idx, benign_directions=sample_n_from_csv(**files[0], n=400)
                if name.startswith("kitsune") and epoch != "1":
                    continue
                model_name=f"{dataset}_{name}_{epoch}"
                print(model_name)
                nids_model=get_nids_model(model_name, threshold_key="opt_t")
                
                if not test_mode and near_threshold:
                    target_idx, start_samples, file_names=reservoir_sample(files, nids=nids_model, n=2000, near_boundary=near_threshold)
                    
                # continue
                # if near_threshold:
                #     eps=0.34349777798961395
                # else:
                #     eps=1.8395914625818583
                eps=0.34349777798961395
                
                csv_f = open(f"exp_csv/adv_detect/{dataset}/{nids_model.name}_{'near_threshold' if near_threshold else 'sample'}_random_dir_0.3.csv", "w")
                metrics=["file","eps","p","idx","score","label", "max_flips","flip_prob","adv_area","adv_prob","mean_turn","max_turn"]
                csv_f.write(",".join(metrics))
                csv_f.write("\n")
                
                if test_mode:
                    write_to_file =False 
                else:
                    write_to_file=csv_f
                for p in norms:
                    adv_detect_exp(nids_model,target_idx, start_samples,file_names, feature_range, eps, p, write_to_file,
                                benign_dir=None)
    
    elif args.command=="plot_res":
        all_results=True
        near_threshold=False
        # eval results
        if all_results:
            df_auc=pd.read_csv(f"exp_csv/all_data/performance.csv")
            near_threshold_rob=pd.read_csv(f"exp_csv/all_data/rob-True.csv")
            all_rob=pd.read_csv(f"exp_csv/all_data/rob-False.csv")
            
            adv_summary=pd.concat([near_threshold_rob.assign(loc='Near'),all_rob.assign(loc='All')])
            adv_summary=adv_summary.set_index(["epoch","NIDS","filtered","new_loss","p","loc"])
            dataset=f"all"
        else:
            df_auc=pd.read_csv(f"exp_csv/performance/{dataset}_eval_results.csv")
            
            results=[]
            common={}
            for name, epoch in itertools.product(ae, epochs):
                if name.startswith("kitsune") and epoch !="1":
                    continue
            # for name, epoch in zip(ae, best_epochs):
                model_name=f"{dataset}_{name}_{epoch}"
                nids_model=get_nids_model(model_name, threshold_key="opt_t", load=False)
                result=pd.read_csv(f"exp_csv/adv_detect/{dataset}/{dataset}_{name}_{epoch}_{'near_threshold' if near_threshold else 'sample'}_random_dir_0.3.csv")  
                result["nids"]=nids_model["abbrev"]
                result["threshold"]=nids_model["threshold"]
                result["MELO"]=np.maximum(result["max_flips"] - result["label"] - 1, 0)
                result["vul"]=np.where(result["MELO"]>0, True, False)
                results.append(result)
                if name.startswith("autoencoder_relu"):
                    vul_points=result[result["vul"]]
                    if epoch not in common.keys():
                        common[epoch]=set(zip(vul_points["file"], vul_points["idx"], vul_points["p"]))
                    else:
                        common[epoch]=common[epoch].intersection(set(zip(vul_points["file"], vul_points["idx"], vul_points["p"])))
            # for i,j in common.items():
            #     print(i, j)
            df=pd.concat(results)
            
            df[["NIDS","epoch"]]=df["nids"].str.rsplit("-", n=1, expand=True)
            df["filtered"]=df["NIDS"].str.endswith("F0.2")
            df["new_loss"]=df["NIDS"].str.endswith("D")
            df["NIDS"].replace({"F0.2":"", "D$":""},inplace=True, regex=True)
            
            df["benign"]=np.where(df["file"]=="Cam_1",True, False)
            df["epoch"]=df["epoch"].astype("int64")
            df["p"]=df["p"].astype(str)
            df.drop("file", axis=1, inplace=True)
            
            tmp=[df]
            for e in [20,40,60,80,100]:
                tmp.append(df[df["NIDS"]=="Kit"].assign(epoch=e))
            df=pd.concat(tmp, ignore_index=True)
            
            adv_summary=df.groupby(["epoch","NIDS","filtered","new_loss","p"]).agg({"max_turn":"mean","MELO":"mean","mean_turn":"mean", "vul":"sum", "adv_prob":"mean"})
            adv_summary=adv_summary.join(df.groupby(["epoch","NIDS","filtered","new_loss","p"]).size().to_frame(name="counts"))
            adv_summary.rename({"max_turn":"MLE"},axis=1,inplace=True)
            adv_summary.rename(index={"1.0":"1","2.0":"2","inf":"inf"},inplace=True)
            # adv_summary["rob_count"]=adv_summary["counts"]*(1-adv_summary["robust"])
            
        df_auc.rename(columns={"nids":"NIDS"}, inplace=True)
        df_auc["filtered"]=df_auc["NIDS"].str.endswith("F0.2")
        df_auc["new_loss"]=df_auc["NIDS"].str.endswith("D")
        df_auc["NIDS"].replace({"F0.2":"", "D$":""},inplace=True, regex=True)
        

        df_auc=df_auc.set_index(["NIDS","epoch","filtered","new_loss"])
        # print(df_auc.to_csv())
        # df_auc=df_auc.reset_index()
        
        
        df_auc_melted=pd.melt(df_auc.query("(filtered==False) & (new_loss==False)"), value_vars=["PR-AUC","opt_f1"],ignore_index=False)
        fig=sns.relplot(data=df_auc_melted, x="epoch", y="value", hue="NIDS", hue_order=["AE","AE_R","DAE","AE_25","Kit"], kind="line", col="variable",height=2, aspect=1, facet_kws={"sharey":'none'}, errorbar=None)
        sns.move_legend(fig, loc="lower center", ncol = 5, bbox_to_anchor = (0.42,0.95),handlelength=1, columnspacing=1.5)

        fig.set_xlabels("Epoch")
        fig.axes[0,0].set_title("Average Precision")
        fig.axes[0,1].set_title("Optimal F1")
        fig.set_ylabels("")
        fig.tight_layout()
        fig.savefig(f"exp_figs/meta_plots/{dataset}_performance.pdf")


        filtered_df=df_auc.query("(NIDS in ['AE_R','AE','DAE','Kit']) & (new_loss==False)").droplevel("new_loss").drop(columns="dataset")
        filtered_df=filtered_df.query("filtered==True").droplevel("filtered").subtract(filtered_df.query("(filtered==False)").droplevel("filtered"))
        
        new_loss_df=df_auc.query("(NIDS in ['AE_R','AE','DAE']) & (filtered==False)").droplevel("filtered").drop(columns="dataset")
        new_loss_df=new_loss_df.query("new_loss==True").droplevel("new_loss").subtract(new_loss_df.query("(new_loss==False)").droplevel("new_loss"))

        comp_df=pd.concat([new_loss_df.assign(method='DLF'),filtered_df.assign(method='FSP')])

        comp_df=comp_df.set_index(["method"], append=True)
        df_auc_melted=pd.melt(comp_df, value_vars=["PR-AUC","opt_f1"],ignore_index=False)
        df_auc_melted=df_auc_melted.reset_index()
        df_auc_melted["col_var"]=df_auc_melted["variable"]+" "+df_auc_melted["method"]
        fig=sns.relplot(data=df_auc_melted, x="epoch", y="value", hue="NIDS",kind="line", col="variable", row="method",
                        height=1.5, aspect=1.3, errorbar=None,facet_kws={"sharey":'row'})
        fig.set_xlabels("Epoch")
        fig.set_ylabels("")
        for (row_key, col_key),ax in fig.axes_dict.items():
            if col_key=="PR-AUC":
                col_key=r"$\delta$AP"
            else:
                col_key=r"$\delta$OF1"
            ax.set_title(f"{row_key} {col_key}")
        for ax in fig.axes.flat:
            ax.axhline(y=0, color='black', linestyle='--')
        sns.move_legend(fig, loc="lower center", ncol = 4, bbox_to_anchor = (0.42,0.95),handlelength=1, columnspacing=1.5)
        
        
        fig.tight_layout()
        fig.savefig(f"exp_figs/meta_plots/{dataset}_performance_comp.pdf")
       
        #"row":"filtered", "col":"new_loss"
        plot_kwargs={"x":"epoch","y":"MLE", "row":"loc", "hue":"NIDS", "col":"p","height":1.2,
                     "aspect":1, "errorbar":None,  "facet_kws":{"sharey":"row"}}
        
        
        adv_detect_melt=adv_summary.query("(filtered==False) & (new_loss==False)")
        fig=sns.relplot(data=adv_detect_melt, kind="line", hue_order=["AE","AE_R","DAE","AE_25","Kit"], **plot_kwargs)
        fig.set_xlabels("Epoch")
        fig.set_ylabels("")
        sns.move_legend(fig, loc="lower center", ncol = 5, bbox_to_anchor = (0.42,0.95),handlelength=1, columnspacing=1.5)
                
        for i, row_name in enumerate(["Near","All"]):
            for j, col_name in enumerate([r"$l_1$",r"$l_2$",r"$l_\infty$"]):
            
                fig.axes[i,j].set_title(f"{col_name} {row_name}")
            
            
        fig.tight_layout()
        fig.savefig(f"exp_figs/adv_detect/{dataset}_{'near_threshold' if near_threshold else 'sample'}.pdf")
        
        
        near_threshold_rob=near_threshold_rob.set_index(["epoch","NIDS","filtered","new_loss","p"])
        filtered_df=near_threshold_rob.query("(NIDS in ['AE_R','AE','DAE','Kit']) & (new_loss==False)").droplevel("new_loss").drop(columns="dataset")
        filtered_df=filtered_df.query("filtered==True").droplevel("filtered").subtract(filtered_df.query("(filtered==False)").droplevel("filtered"))
        
        new_loss_df=near_threshold_rob.query("(NIDS in ['AE_R','AE','DAE']) & (filtered==False)").droplevel("filtered").drop(columns="dataset")
        new_loss_df=new_loss_df.query("new_loss==True").droplevel("new_loss").subtract(new_loss_df.query("(new_loss==False)").droplevel("new_loss"))

        comp_df=pd.concat([new_loss_df.assign(method='DLF'),filtered_df.assign(method='FSP')])
        
        fig=sns.relplot(data=comp_df, x="epoch", y="MLE", hue="NIDS",kind="line", col="p", row="method",
                        height=1.5, aspect=1, errorbar=None,facet_kws={"sharey":'row'})
        
        fig.set_ylabels("")
        sns.move_legend(fig, loc="lower center", ncol = 5, bbox_to_anchor = (0.42,0.95),handlelength=1, columnspacing=1.5)
        for (row_key, col_key),ax in fig.axes_dict.items():
            if col_key==1:
                col_key=r"$l_1$"
            elif col_key==2:
                col_key=r"$l_2$"
            else:
                col_key=r"$l_\infty$"
            ax.set_title(f"{row_key} {col_key} $\delta$MLE")
        for ax in fig.axes.flat:
            ax.axhline(y=0, color='black', linestyle='--')
        fig.tight_layout()
        fig.savefig(f"exp_figs/adv_detect/{dataset}_rob_comp.pdf")
        
        
        
    elif args.command=="adv_dist":
        measure_feature_distance("uq","ACK_Flooding","autoencoder_0.1_10_3_False_pso0.5",854685, scaler=scaler)
        
    elif args.command=="train_nids":
        
        training_datasets=[False, True]
        if not os.path.exists(f"../models/{dataset}/"):
            os.mkdir(f"../models/{dataset}/")
        save_epoch=[1]
        
        with open("configs/nids_models.json","r") as f:
            nids_db=json.load(f)

        for filtered in training_datasets: 
            files=[f"{dataset}_train{'_filtered_0.2' if filtered else ''}"]
            files=get_files(files)
            model_name=f"kitsune{'_filtered_0.2' if filtered else ''}"
            model_param_temp={
                        "abbrev":f"Kit{'F0.2' if filtered else ''}",
                        "path": f"../models/{dataset}/{model_name}",		            
                        "func_name": "process",
                        "scaler": None,
                        "input_dim":46,
                        "flip_score": False,
                        "perc":0.8,
                        "epochs":max(save_epoch),
                        "batch_size":1,
                        "save_type": "pkl"}
            
            train_nids(files[0], model_param_temp, save_epoch)
            
            for epoch in save_epoch:
                model_param=dict(model_param_temp)
                model_param["abbrev"]+=f"-{epoch}"
                model_param["path"]+=f"_{epoch}.pkl"
                nids_db[f"{dataset}_{model_name}_{epoch}"]=model_param 
            
        with open("configs/nids_models.json","w") as f:    
            json.dump(nids_db, f, indent=4)
        
        
    elif args.command=="find_threshold":
        
        with open("configs/nids_models.json","r") as f:
            nids_db=json.load(f)
            
        files=[f"{dataset}_train",f"{dataset}",f"{dataset}_ATK"]
        files=get_files(files)
        
        if not os.path.exists(f"exp_figs/meta_plots/pr_curve/{dataset}"):
            os.mkdir(f"exp_figs/meta_plots/pr_curve/{dataset}")
            os.mkdir(f"exp_figs/meta_plots/anomaly_scores/{dataset}")
            
        
        with open(f"exp_csv/performance/{dataset}_eval_results.csv","w") as results:
            results.write("nids,epoch,PR-AUC,opt_t,opt_f1,precision,recall,real_t,real_f1\n")
            for name, epoch in tqdm(itertools.product(ae, epochs)):
                if name.startswith("kitsune") and epoch != "1":
                    continue
                
                model_name=f"{name}_{epoch}"
                print(dataset,model_name)
                model_param=nids_db[f"{dataset}_{model_name}"]     
                nids_model=get_nids_model(f"{dataset}_{model_name}")
                
                thresholds=eval_nids(dataset,files,[0,0,1], nids_model, plot=True, full_data=False, scaler=None, file_handle=results)
                
                model_param["thresholds"]=thresholds
                nids_db.update({f"{dataset}_{model_name}":model_param})
                
        with open("configs/nids_models.json","w") as f:
            json.dump(nids_db, f, indent=4)
                
    elif args.command=="plot_adr":
        plot_adr("exp_csv/adv_detect/ADR.csv")
        