import numpy as np 
from adv_based_vis import feature_to_2d, sample_n_from_csv,get_closest_benign_sample, ln_distance, gradient_estimate, angle
from helper import *
import pickle
from tqdm import tqdm 
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from plotly.subplots import make_subplots
from shap_vis import *
import scienceplots
plt.style.use('science')


def get_plane_coord(boundary_file=None, feature_range=None, plot_range=None, nids_model=None, plane="boundary", x_0=None):
    
    if x_0 is None:
        boundary_path = np.genfromtxt(boundary_file, delimiter=",")
        if boundary_path.ndim==1:
            return False, False, False, False
        x_0 = boundary_path[np.newaxis, 0]    
    
    boundary_result=None

    if isinstance(plane, np.ndarray):
        A=plane.T
        
    elif plane=="stochastic":
        A=np.random.uniform(-1,1,(x_0.shape[1],2))
        A/=np.linalg.norm(A, axis=-1, keepdims=True)
   
    else:
        A=boundary_path[-2:].T   
    
        boundary_path=boundary_path[:-2]
        boundary_result = feature_to_2d(
            x_0, boundary_path, A, feature_range=feature_range)
    
    
    len_A=np.linalg.norm(A*feature_range[:,np.newaxis],axis=0)/np.linalg.norm(feature_range)
        
    if plot_range is None:
        no_end=boundary_result[:,:-1]
        dir1 = np.linspace(np.min(
        no_end[0])-np.ptp(no_end[0])*0.1, np.max(no_end[0])+np.ptp(no_end[0])*0.1, 201)
        dir2 = np.linspace(np.min(
            no_end[1])-np.ptp(no_end[1])*0.1, np.max(no_end[1])+np.ptp(no_end[1])*0.1, 201)
    else:
        dir1=np.linspace(plot_range[0][0], plot_range[0][1], plot_range[0][2])
        dir2=np.linspace(plot_range[1][0], plot_range[1][1], plot_range[1][2])
    
    xv, yv = np.meshgrid(dir1, dir2)

    coord_mat = np.dstack([xv, yv])

    
    if feature_range is None:
        input_val = x_0+np.einsum("ijk,lk->ijl", coord_mat, A)
    
    else:
        input_val = x_0+np.einsum("ijk,lk,l->ijl", coord_mat, A, feature_range)
    
    input_val = input_val.reshape([-1, A.shape[0]])

    f_val = nids_model.predict(input_val)
    f_val = f_val.reshape(xv.shape)
    
            
    return boundary_result, xv, yv, f_val, len_A
    

def average_db(boundary_folders, feature_range):
    for boundary_folder in boundary_folders:
        x_coord=[]
        y_coord=[]
        last_idx=-1 
        if not boundary_folder.endswith("grad_attrib") or boundary_folder.endswith("random"):
            last_idx-=1
        for boundary in tqdm(os.listdir(boundary_folder)):

            coords=get_plane_coord(f"{boundary_folder}/{boundary}", feature_range)
            #first idx is start and last 2 are anchor and end points
            x_coord.append(coords[0,1:last_idx])
            y_coord.append(coords[1,1:last_idx])
        x_coord=np.hstack(x_coord)
        y_coord=np.hstack(y_coord)
        
        z, x_edges, y_edges=np.histogram2d(x_coord, y_coord, bins=200)
        fig = go.Figure(go.Heatmap(
        z= np.log1p(z),
        x=x_edges, 
        y=y_edges
        ))
    
        # fig=px.scatter(x=x_coord, y=y_coord)
        fig.write_html(f"exp_figs/db_vis/{boundary_folder.split('/')[-1]}/average_db.html")

def plane_std(boundary_folders, feature_range, nids_model):
    for boundary_folder in boundary_folders:
        file_name=boundary_folder.split('/')[-1]
        idx=file_name.find("_Cam_1_")
        with open(f"exp_csv/{file_name[6:idx+6]}_ext.csv","a") as f:
            for boundary in tqdm(os.listdir(boundary_folder)):

                _,_,_,fval=get_plane_coord(f"{boundary_folder}/{boundary}", feature_range, 
                                        return_contour=True, nids_model=nids_model, plot_range=[[-4,4,101],[-4,4,101]])
                
                f.write(",".join([file_name, boundary[:-4], str(fval.std()),str(fval.mean()), str(fval[50,50])]))
                f.write("\n")
            


def plot_plane_metrics(name, threshold, start):
    data=pd.read_csv(f"exp_csv/db_characteristics/{name}_{threshold:.3f}_bt_results_{start}.csv", delimiter=",")
    
    data["area"]=data["area"].fillna(0)
    
    
        
    data[["device","nids","plane"]]=data["run_name"].str.split("/",expand=True)
    
    data[["nids","epoch","threshold"]]=data["nids"].str.rsplit("_",n=2,expand=True)
    
    data["epoch"]=data["epoch"].astype("int32")
    data["complexity"]=data["discontinuous"]/data["distance"]
    data["ad_ratio"]=data["area"]/data["distance"]
    data["incomplete"]=data["complete"]!=2
    data["unbounded"]=np.where((data["complete"]==2)&~data["enclosed"], True, False)
    data["target"]=start
    data["label"]=data["score"]>threshold
    
    replace_dict={"42_.*grad_attrib-ae":"IG-AE","42_.*grad_attrib-pca":"IG-PCA","42_.*_random":"Random","42.*Cam_1":"Interpolated",
                "_filtered_0.1":"F0.1",
                "kitsune":"Kit","denoising_autoencoder_sigmoid_2":"DAE","autoencoder_sigmoid_25":"AE_25","autoencoder_sigmoid_2":"AE","autoencoder_relu_2":"AE_R"}
    
    data=data.replace(replace_dict,regex=True)
    init=data.groupby(["nids","plane","target","epoch"]).agg('mean')["init"]

    unbounded=data[~data["incomplete"]&~data["label"]].groupby(["nids","plane","target","epoch"]).agg('mean')["unbounded"]

    # data[data["smoothness"]==np.inf]=np.nan
    # ["nids","plane","target","complexity","area","epoch","incomplete","init_dist","unbounded","ad_ratio","discontinuous"]
    aggregated=data[~data["incomplete"]].groupby(["nids","plane","target", "epoch"]).agg('mean')
    
    aggregated.update(init)
    aggregated.update(unbounded)


    # [["nids","plane","target","init_dist","unbounded","epoch","ad_ratio","discontinuous"]]
    # init_dist=data.groupby(["nids","plane","target","epoch","unbounded"]).agg("mean")
    
    return aggregated
    
def as_between_benign_test(benign_file, total_rows, nids_model, scaler):
    
    num_samples=11
    # traffic_ds = get_dataset(benign_file, 128,
    #                          scaler=None, frac=1, read_with="tf", dtype="float64",
    #                          seed=0, skip_header=True, shuffle=True)
    
    _,traffic_ds=sample_n_from_csv(benign_file, n=1000, ignore_rows=1, total_rows=total_rows, seed=42)
    traffic_ds=np.array_split(traffic_ds, 10)
    counter=0
    positive=0
    
    eps=[1e-2,0.1,0.3,0.5]
    num_samples=[11,101,101,101]
    with open(f"exp_csv/profile_detect/{nids_model.name}_pairwise_nearest.csv","w") as f:
        f.write("eps,distance,monotonic,within_t\n")
        for e, n in zip(eps,num_samples):
            for data in tqdm(traffic_ds):
                # data=data.numpy()
                idx=np.arange(counter, counter+data.shape[0])+1
                nearest_sample, distance, c_idx=get_closest_benign_sample(benign_file, data, transform_func=scaler.transform, eps=e)
                result, within_t=monotonic_between_sample(data, nearest_sample,nids_model, n)
                print(distance.shape, result.shape)   
                np.savetxt(f, np.hstack([np.full((result.shape[0],1),e),distance.T,result[:,np.newaxis],within_t[:,np.newaxis]]), delimiter=",") 
                counter+=data.shape[0]
                positive+=np.sum(result)
            
    

def monotonic_between_sample(x,y,nids_model, num_samples):
    samples=np.linspace(x,y,num_samples)
    samples=samples.reshape([-1,100])
    scores=nids_model.predict(samples)
    scores=scores.reshape([num_samples,x.shape[0]])
    diff=np.diff(scores, axis=0).T
    result=np.where(diff[:,0]>0, np.all(diff>0, axis=1), np.all(diff<0, axis=1))
    return result, np.max(scores, axis=0)<nids_model.threshold

def as_between_random_bengin_test(benign_file, nids_model, scaler, feature_range):
    n_random=3
    seed=42
    random_datasets=[get_dataset(benign_file, 1024,
                             scaler=None, frac=1, read_with="tf", dtype="float64",
                             seed=0, skip_header=True, shuffle=False)]
    for i in range(n_random):
        random_datasets.append(get_dataset(benign_file, 1024,
                             scaler=None, frac=1, read_with="tf", dtype="float64",
                             seed=seed+i, skip_header=True, shuffle=True))
    
    positive_count=0
    
    counter=0
    with open(f"exp_csv/profile_detect/{nids_model.name}_pairwise_random.csv","w") as f:
        f.write("distance,monotonic,within_t\n")
        for data in tqdm(zip(*random_datasets)):
            data=list(data)
            for i in range(n_random+1):
                data[i]=data[i].numpy()
            
            for i in range(n_random):
                result, within_t=monotonic_between_sample(data[0], data[i+1],nids_model, 11)
                distance=ln_distance(scaler.transform(data[0]), scaler.transform(data[i+1]), 2)
                
                np.savetxt(f, np.hstack([distance,result[:,np.newaxis], within_t[:,np.newaxis]]), delimiter=",")
                positive_count+=np.sum(result)
            
            counter+=data[0].shape[0]
    print(positive_count/counter)

def plot_monontic_results(nids_model_name, neighbour_type):
    df = pd.read_csv(f"exp_csv/profile_detect/{nids_model_name}_pairwise_{neighbour_type}.csv")
    
    fig, ax=plt.subplots(3, 1, figsize=(5,5))
    sns.violinplot(data=df, x="monotonic",split=True,
                    y="distance", inner=None, cut=0, ax=ax[1])
    sns.violinplot(data=df, x="within_t",y="distance", ax=ax[0])
    sns.countplot(data=df, x="within_t",hue="monotonic", ax=ax[2])
    print(sum(df["within_t"])/len(df))
    # ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # fig.set(ylim=(0,10))
    fig.tight_layout()
    
    fig.savefig(f"exp_figs/profile_detect/{nids_model_name}_pairwise_random.png")
    # print(df.groupby("eps").sum("monotonic"))

def feature_distribution(boundary_folders,feature_names):
    data=[]
    names=[]
    for folder_name, boundary_folder in boundary_folders.items():
        start_sample=[]
        for boundary in tqdm(os.listdir(boundary_folder)):
            boundary_path = np.genfromtxt(f"{boundary_folder}/{boundary}", delimiter=",")
            start_sample.append(boundary_path[np.newaxis,0])
            names.append(folder_name)
        start_sample=np.vstack(start_sample)
        print(folder_name)
        print(np.std(start_sample, 0))
        data.append(start_sample)
    data=np.vstack(data)
    
    names=np.array(names)    
    df=pd.DataFrame(data, columns=feature_names)
    df["folder"]=names
    fig, ax_new = plt.subplots(10,10, sharey=False,figsize=(25,25))
    
    df.boxplot(by="folder", ax=ax_new)
    fig.tight_layout()
    fig.savefig(f"exp_figs/db_vis/feature_distribution.png")



def eval_plane(boundary_folders, nids_model, feature_range, eps=[0.01]):
    filename="exp_csv/db_characteristics/accuracy.csv"
    
    file_exists = os.path.isfile(filename)
    accuracy_file=open(filename,"a")
    if not file_exists:
        accuracy_file.write("run_name,idx,score,eps,accuracy,recon_error,recon_error_normalised\n")

    for boundary_folder in tqdm(boundary_folders):
        run_name="/".join(boundary_folder.split('/')[2:])
        for boundary in tqdm(os.listdir(boundary_folder)):
            boundary_path = np.genfromtxt(f"{boundary_folder}/{boundary}", delimiter=",")
            x_0,_,A=np.split(boundary_path,[1,-2])
            
            for e in eps:
                ori_score=nids_model.predict(x_0)[0]
                directions=feature_range * sample_d_sphere(1000,100)
                
                
                directions/=np.linalg.norm(directions, axis=1,keepdims=True)
                
                random_background=x_0 + feature_range*directions*e
                
                
                true_val=nids_model.predict(random_background)
                projected_coord,_,_,_=np.linalg.lstsq(A.T,((random_background-x_0)/feature_range).T,rcond=None)
                
                projected_coord/=np.linalg.norm(projected_coord, axis=0)/e 
                
                projected_feature=x_0+np.einsum("ji,jk,k->ik",projected_coord,A, feature_range)
                
                
                recon_error=ln_distance(projected_feature, random_background, 2)
                recon_error2=ln_distance(projected_feature/feature_range, random_background/feature_range, 2)
                
                pred_val=nids_model.predict(projected_feature)
                
                
                accuracy_file.write(",".join([ run_name,boundary.split(".")[0], str(ori_score) ,str(e), str(np.mean(np.abs(true_val-pred_val))), str(np.mean(recon_error)), str(np.mean(recon_error2))])) 
                accuracy_file.write("\n")

            

def plot_overlapping_heatmap(boundary_folders, nids_model, feature_range, threshold, normalize=True, files=None, ):
    for boundary_folder in tqdm(boundary_folders):
        
        x_vals=[]
        y_vals=[]
        z_vals=[]
        fig = make_subplots(rows=1, cols=2)
        end=[]
        start=[]
        if boundary_folder.endswith("Cam_1"):
            tangent=[]
        else:
            tangent=None
        for boundary in tqdm(os.listdir(boundary_folder)):
            if files:
                if boundary!=files:
                    continue
            boundary_result, xv, yv, zv=get_plane_coord(f"{boundary_folder}/{boundary}",feature_range, return_contour=True,)
            if boundary_result is False:
                continue
            
            if normalize:
                boundary_result[0]=(boundary_result[0]-xv.min())/np.ptp(xv)*2-1
                boundary_result[1]=(boundary_result[1]-yv.min())/np.ptp(yv)*2-1
                xv=(xv-xv.min())/np.ptp(xv)*2-1
                yv=(yv-yv.min())/np.ptp(yv)*2-1
           
            x_vals.append(xv.flatten())
            y_vals.append(yv.flatten())
            z_vals.append(zv.flatten())
            end.append(boundary_result[:,-1])
            start.append(boundary_result[:,0])
            if tangent is not None:
                tangent.append(boundary_result[:,-2])
            
        x_vals=np.hstack(x_vals)
        y_vals=np.hstack(y_vals)
        z_vals=np.hstack(z_vals)

        start=np.vstack(start)
        end=np.vstack(end)

        if tangent is not None:
            tangent=np.vstack(tangent)
             
        median, x_edge, y_edge, _=scipy.stats.binned_statistic_2d(x_vals, y_vals, z_vals, statistic='median',bins=100)

        fig.add_trace(go.Contour(
        z= median.T,
        x=x_edge, 
        y=y_edge,
        connectgaps=True,
        line_smoothing=0.85,
        contours_coloring='lines',
        colorscale="greys",
        line_width=2,
        contours=dict(
                    showlabels=True,  # show labels on contours
                    labelfont=dict(  # label font properties
                        size=12,
                        color='white',
                    ),
                    
                    start=0,
                    end=threshold*2,
                    size=(threshold)/5.
                    
                )
        ), row=1, col=1)
        fig.add_trace(go.Scattergl(x=start[:,0], y=start[:,1], mode="markers",marker_symbol="circle"), row=1, col=1)
        
        fig.add_trace(go.Scattergl(x=end[:,0], y=end[:,1],mode="markers", marker_symbol="star"), row=1, col=1)
        if files:
            fig.add_trace(go.Scattergl(x=boundary_result[0], y=boundary_result[1],mode="markers", marker_symbol="x"), row=1, col=1)
        
        if tangent is not None:
            fig.add_trace(go.Scattergl(x=tangent[:,0], y=tangent[:,1], mode="markers",marker_symbol="triangle-up"), row=1, col=1)
        
        fig.add_trace(go.Heatmap(z= median.T,
        x=x_edge, 
        y=y_edge,connectgaps=True), row=1, col=1)
        
        std, x_edge, y_edge, _=scipy.stats.binned_statistic_2d(x_vals, y_vals, z_vals, statistic=np.min,bins=100)

        fig.add_trace(go.Contour(
        z= median.T,
        x=x_edge, 
        y=y_edge,
        connectgaps=True,
        line_smoothing=0.85,
        contours_coloring='lines',
        colorscale="greys",
        line_width=2,
        contours=dict(
                    showlabels=True,  # show labels on contours
                    labelfont=dict(  # label font properties
                        size=12,
                        color='white',
                    ),
                    
                    start=0,
                    end=threshold*2,
                    size=(threshold)/5.
                    
                )
        ), row=1, col=2)
        fig.add_trace(go.Scattergl(x=np.array([0]), y=np.array([0]), mode="markers",marker_symbol="circle"), row=1, col=2)

        fig.add_trace(go.Scattergl(x=end[:,0], y=end[:,1],mode="markers", marker_symbol="star"), row=1, col=2)
        
        if tangent is not None:
            fig.add_trace(go.Scattergl(x=tangent[:,0], y=tangent[:,1], mode="markers",marker_symbol="triangle-up"), row=1, col=2)
        
        fig.add_trace(go.Heatmap(z= std.T,
        x=x_edge, 
        y=y_edge,connectgaps=True), row=1, col=2)
        # fig=px.scatter(x=x_coord, y=y_coord)
        print(boundary_folder.split('/')[-1])
        if files:
            fig.write_html(f"exp_figs/db_vis/{'/'.join(boundary_folder.split('/')[2:])}/{files.split('.')[0]}_{'raw' if not normalize else ''}.html")
        
        else:
            fig.write_html(f"exp_figs/db_vis/{'/'.join(boundary_folder.split('/')[2:])}/average_contour_{'raw' if not normalize else ''}.html")

def compare_db_contour(nids, plane, idx, start,feature_range,n_cols=None, titles=None, plot_range=None, backend="sns"):
    
    n_plots=len(nids)*len(plane)*len(idx)
    
    name=[]
    if len(nids)==1:
        name.append(nids[0].name)
        if n_cols is None:
            n_cols=len(plane)
    if len(plane)==1:
        if isinstance(plane[0], np.ndarray):
            name.append("custom_plane")
        else:
            name.append(plane[0])

        if n_cols is None:
            n_cols=len(idx)
    if len(idx)==1:
        if isinstance(idx[0], np.ndarray):
            name.append("custom_x_0")
        else:
            name.append(str(idx[0]))
        if n_cols is None:
            n_cols=len(nids)
    
    if backend=="plotly":
        fig = make_subplots(rows=n_plots//n_cols, cols=n_cols, horizontal_spacing = 0.01, vertical_spacing=0.08)
    elif backend=="sns":
        fig, axes = plt.subplots(n_plots//n_cols, n_cols, squeeze=False, figsize=(n_cols*1.5,n_plots//n_cols*1.5))
        
    row=1
    col=1
    for n in nids:
        for p in plane:
            for i in idx:
                
                if isinstance(i, np.ndarray):
                    boundary_result, xv, yv, zv, len_A=get_plane_coord(None, feature_range=feature_range, plot_range=plot_range, nids_model=n, plane=p, x_0=i)

                    
                else:
                    if isinstance(p, np.ndarray) or p=="stochastic":
                        path=f"../adversarial_data/Cam_1/{n.name}_{n.threshold:.3f}/42_{start}_random/{i}.csv"
                    else:
                        path=f"../adversarial_data/Cam_1/{n.name}_{n.threshold:.3f}/42_{start}_{p}/{i}.csv"
                    boundary_result, xv, yv, zv, len_A=get_plane_coord(path, feature_range, plot_range=plot_range, nids_model=n, plane=p)
                
                if backend=="plotly":
                    fig.add_trace(go.Contour(
                    z=zv,
                    x=xv[0], 
                    y=yv[:,0],
                    connectgaps=True,
                    line_smoothing=0.85,
                    # contours_coloring='lines',
                    # colorscale="greys",
                    line_width=2,
                    opacity=0.5,
                    contours=dict(
                                showlabels=True,  # show labels on contours
                                labelfont=dict(  # label font properties
                                    size=12,
                                    color='white',
                                ),
                                # start=0,
                                # end=n.threshold*2,
                                # size=n.threshold/5
                            )
                    ), row=row, col=col)
                    
                    if boundary_result is not None:
                        fig.add_trace(go.Scattergl(x=boundary_result[0], y=boundary_result[1],mode="markers", 
                                                marker_symbol="circle", marker_color="green",
                                                opacity=0.5),
                                    row=row, col=col)
            
                
                    fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.3, showarrow=False,
                    text=f"{n.name} {p} {i} <br> |v1| = {len_A[0]:.3e} |v2|={len_A[1]:.3e}", row=row, col=col)
                else:
                    n_levels=5
                    levels=np.linspace(0,n.threshold*2,n_levels)
                    # levels=None
                    axes[row-1, col-1].contourf(xv,yv, zv, levels, cmap="coolwarm", extend="both")
                    c=axes[row-1, col-1].contour(xv,yv, zv, levels, colors="black")

                    c.collections[len(levels)//2].set_linestyle('--')
                    if boundary_result is not None:
                        axes[row-1, col-1].scatter(x=boundary_result[0], y=boundary_result[1], s=10, color="red")
                    else:
                        
                        axes[row-1, col-1].scatter(x=[0,-4.022,-1.987], y=[0,0,-3.133], s=10, marker="x", color="black")
                        
                        
            
                    # axes[row-1, col-1].clabel(c, c.levels, inline=True, fmt='%2.3f')
                    axes[row-1, col-1].axis('off')
                    if titles:
                        axes[row-1, col-1].set_title(titles[(row-1)*n_cols+(col-1)])
                    else:
                        if isinstance(p, np.ndarray):
                            p="custom_plane"
                        if isinstance(i, np.ndarray):
                            i="custom_point"
                     
                        abbrev, epoch=n.abbrev.split("-")
                        if col==1:
                            axes[row-1, col-1].text(-8.5, -1, f"{abbrev}", ha='center', va='center', rotation='vertical')
                            # axes[row-1, col-1].set_ylabel()
                        if row==1:
                            axes[row-1, col-1].set_title(f"Epoch: {epoch}")
                        
                    
                if col==n_cols:
                    row+=1 
                    col=1
                else:
                    col+=1
    if backend=="plotly":
        fig.write_html(f"exp_figs/db_vis/{start}_{'_'.join(name)}.html")
    elif backend=="sns":
        # fig.tight_layout()
        plt.subplots_adjust(hspace=0.01, wspace=0.01)

        # fig.savefig(f"exp_figs/db_vis/{start}_{'_'.join(name)}.pdf")
        fig.savefig(f"exp_figs/db_vis/example_db.pdf")
if __name__=="__main__":
    
   
    mtd_model_path = f"../../mtd_defence/models/uq/mtd/Cam_1/fm0_mm1_am20"
    
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
    
    time_windows={"5":"0.1s", "3":"0.5s","1":"1.5s","0.1":"10s","0.01":"1m"}
    relations={"MI":"ps from srcMAC&srcIP","jit":"iat srcIP-dstIP","H":"ps srcIP-dstIP","Hp":"ps srcIP&port-dstIP&port"}
    stats={"weight":"freq of","mean": "mean","std":"std of", "magnitude":"bi-directional mean of", "radius":"bi-directional var of", "covariance": "relationship between","pcc":"scaled relationship between"}
    
    modified_feature_names=[]
    for i in feature_names:
        components=i.split("_")
        new_name=f"{stats[components[-1]]} {relations[components[-3]]} in {time_windows[components[-2]]}"
        modified_feature_names.append(new_name)
    
    scaler_path = f"../../mtd_defence/models/uq/autoencoder/Cam_1_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    feature_range = (scaler.data_max_ - scaler.data_min_)
    
    with open("configs/nids_models.json","r") as f:
        nids_db=json.load(f)
    
    #"grad_attrib-ae","grad_attrib-pca","random","Cam_1"
    custom_point=np.array([ 6.759687908136286616e+00,1.959654457503420133e+02,1.359970054612291278e+03,9.221524139872636638e+00,1.963892593548756054e+02,1.597543971709907055e+03,1.649651319535458782e+01,1.975996718721906120e+02,1.513156605131189281e+03,2.684120715488597142e+01,2.972922316065599375e+02,1.117726266771917144e+05,6.304284075722396352e+01,5.356012661378860003e+02,3.016186231829038006e+05,6.759687908136288392e+00,1.959654457503420133e+02,1.359970054612291278e+03,2.931321475514587860e+02,1.359970054612291278e+03,-7.177085319680995152e-14,0.000000000000000000e+00,9.221524139872636638e+00,1.963892593548756054e+02,1.597543971709907055e+03,2.934156457824915947e+02,1.597543971709907055e+03,-8.685299985430647203e-08,0.000000000000000000e+00,1.649651319535458782e+01,1.975996718721906120e+02,1.513156605131189281e+03,2.942271746864952888e+02,1.513156605131189281e+03,-6.773587253047200218e-02,0.000000000000000000e+00,2.684120715488597853e+01,2.972922316065599375e+02,1.117726266771916999e+05,3.686552420949004727e+02,1.117726266771916999e+05,-5.684975744329145897e+01,-2.057181147266620158e+01,6.304284075722397063e+01,5.356012661378860003e+02,3.016186231829038006e+05,5.786603798736363160e+02,3.016186231838454260e+05,-2.035814991314855149e+01,-4.269932159031730040e-02,6.167283610443090147e+00,6.395053280048345057e-01,9.623092729673480505e+01,8.262268774155650419e+00,2.067675683326615221e+00,3.515206266915115521e+02,1.493130118641908233e+01,5.473733994390708624e+00,9.414565605558116204e+02,2.194104242408295846e+01,7.604530517252986677e+00,1.297993496269323487e+03,3.115375186308260425e+01,8.049102584613308409e+00,1.368156295954971711e+03,6.167283610443090147e+00,1.989999999999999716e+02,1.455191522836685181e-11,2.966496249786943054e+02,1.455191522836685181e-11,0.000000000000000000e+00,0.000000000000000000e+00,8.262268774155650419e+00,1.990000000000000000e+02,7.275957614183425903e-12,2.966496249786943054e+02,7.275957614183425903e-12,0.000000000000000000e+00,0.000000000000000000e+00,1.493130118641908055e+01,1.990000000000000000e+02,1.455191522836685181e-11,2.966496249786943054e+02,1.455191522836685181e-11,0.000000000000000000e+00,0.000000000000000000e+00,2.194097823788843016e+01,1.990000000000000284e+02,1.455191522836685181e-11,2.966496249786836188e+02,2.057951587445997463e-11,0.000000000000000000e+00,0.000000000000000000e+00,2.289097686457165892e+01,1.990000000000000284e+02,7.275957614183425903e-12,2.965890552846714172e+02,8.268586423946540365e-02,0.000000000000000000e+00,0.000000000000000000e+00
])
    custom_interpolated_plane=np.array([[-3.104120753757677598e-02,-1.113547121824748631e-01,-1.521698820508303462e-01,-4.487836025284445279e-02,-1.167253979387711665e-01,-1.453520970100220244e-01,-9.471866763064926142e-02,-1.230377549503585410e-01,-1.377229619446707354e-01,-1.763884832431990013e-01,-1.087279065757747326e-01,-8.280369450003567666e-02,-1.938637735128986850e-01,-7.105398001260143648e-02,7.685797533803992616e-03,-3.104120753757687659e-02,-1.113547121824748909e-01,-1.521698820508302352e-01,-9.093177494408775929e-02,-1.521698641564903642e-01,3.325091438459489165e-05,2.037735112422886221e-10,-4.487836025284445279e-02,-1.167253979387712082e-01,-1.453520970100217191e-01,-9.640961452759082284e-02,-1.453489387218211970e-01,9.380932890851067965e-05,1.366931831360628395e-10,-9.471866763064926142e-02,-1.230377549503585272e-01,-1.377229619446708186e-01,-1.030526805283325242e-01,-1.370313885752443006e-01,7.120545018827598757e-05,1.074599520372361115e-10,-1.763884832431991401e-01,-1.087279065757750518e-01,-8.280369450003441378e-02,-9.170244431769476734e-02,-6.868962464988519201e-02,-6.727611195168511601e-05,-1.946267937703110058e-05,-1.938637735129002670e-01,-7.105398001259950747e-02,7.685797533796183759e-03,-6.382879398552379624e-02,6.125110538390740426e-03,1.120906999741807424e-04,-2.432203412798490942e-07,-3.501515494858100513e-02,1.329686149150849408e-05,7.817636773128910370e-07,-5.319113102002086951e-02,4.350899014635067007e-05,2.691501946522601592e-06,-1.011812693485392017e-01,1.155746233643103403e-04,6.828877249936205096e-06,-1.780446833232233184e-01,1.606616268674170640e-04,9.351568074994846531e-06,-1.753850995068677865e-01,1.700434833337647331e-04,9.856238352222360512e-06,-3.247877397929012366e-02,-1.186663641529166341e-01,-1.441713629302764643e-01,-1.022912850603510948e-01,-1.441713459553146670e-01,1.631674164018716066e-05,6.115006174067376702e-04,-5.062837181897870270e-02,-1.218393315686726042e-01,-1.391354315120392482e-01,-1.054380981310696630e-01,-1.391314005606807402e-01,4.289255615967520706e-05,4.748257286241969015e-03,-9.959552346384359378e-02,-1.247701587736677886e-01,-1.352510054541956330e-01,-1.082735228597575539e-01,-1.352470485396683464e-01,2.560684763968078823e-05,4.347657702581948780e-03,-1.770452549365189032e-01,-1.251957588471676319e-01,-1.342596834363519176e-01,-1.085779382687547534e-01,-1.342558096377092780e-01,2.046665358543070412e-05,1.030156905346680174e-03,-1.753494164535765731e-01,-1.261077389960131578e-01,-1.329066185161260072e-01,-1.094953781030399004e-01,-1.329027647139314483e-01,1.961782581225680902e-05,2.576844669985880695e-04
],
        [2.921914031227818009e-02,-1.938126532754977649e-01,9.733228833812561376e-02,3.970957992142449405e-02,-1.906363014524146138e-01,9.315778882906051994e-02,7.261235166014237685e-02,-1.878372759801671954e-01,8.646576956677992642e-02,1.141382759151551157e-01,-1.364372938673012459e-01,-4.379511091677099066e-03,1.231417616894679851e-01,-7.267054046673464274e-02,-1.828428733795088926e-02,2.921914031227819744e-02,-1.938126532754977371e-01,9.733228833812553049e-02,-1.745704403314908004e-01,9.733227689211237932e-02,-2.108928397835949174e-05,-1.292426847619873898e-10,3.970957992142449405e-02,-1.906363014524145860e-01,9.315778882906033953e-02,-1.721913079017497739e-01,9.315576464650170141e-02,-5.949826100697757877e-05,-8.669720548791561597e-11,7.261235166014237685e-02,-1.878372759801671954e-01,8.646576956677995418e-02,-1.705747832653264628e-01,8.603158326445282944e-02,-4.531015574198155168e-05,-6.815612402719671513e-11,1.141382759151551851e-01,-1.364372938673010238e-01,-4.379511091677914386e-03,-1.215357050322238841e-01,-3.633013749717877230e-03,-9.310820141153429148e-05,-1.265439280921248074e-05,1.231417616894689981e-01,-7.267054046673585010e-02,-1.828428733794621591e-02,-6.754296696404329947e-02,-1.457148613605644891e-02,-1.563055477317190048e-04,-2.327133434649995160e-07,3.236808518943470186e-02,-3.250775772699044525e-03,-3.351521434678126276e-04,4.681846255911920052e-02,-3.999666797559747224e-03,-3.481732322886914983e-04,7.980946596495021306e-02,-4.782274666651127083e-03,-3.446470694689990843e-04,1.167172730078154252e-01,-5.135911208157000266e-03,-3.429025823698969534e-04,1.119427653419036855e-01,-5.024547308682804879e-03,-3.317741061378648936e-04,2.977155401789264091e-02,-1.885016344255115717e-01,9.144021662731612043e-02,-1.772225964230935980e-01,9.144020586100197467e-02,-1.034884015733574078e-05,-3.878422717724575703e-04,4.318294369189910192e-02,-1.867760920326098084e-01,8.824619355335405757e-02,-1.754983092258264321e-01,8.824363693560458022e-02,-2.720446382154847936e-05,-3.011566694185841431e-03,7.453435473113358134e-02,-1.862628914630451715e-01,8.578250899781715566e-02,-1.749313151222313956e-01,8.577999933768633289e-02,-1.624105958162776412e-05,-2.757487715072629020e-03,1.140768843477404509e-01,-1.879421883636188118e-01,8.515376624185846954e-02,-1.765248672147962938e-01,8.515130929770636792e-02,-1.298090827089637116e-05,-6.533736566712072726e-04,1.108271748509807864e-01,-1.874567986290998078e-01,8.429558923013003024e-02,-1.760415644017154224e-01,8.429319510837564311e-02,-1.244254202477965387e-05,-1.634355325838238269e-04
]])
    
    #"kitsune_60","autoencoder_relu_2_40","autoencoder_sigmoid_25_100","autoencoder_sigmoid_2_100","denoising_autoencoder_sigmoid_2_100"
    # model_names=["autoencoder_sigmoid_2_100","autoencoder_sigmoid_2_filtered_100","autoencoder_sigmoid_25_100"]
    nids_models=[
    "autoencoder_relu_2","autoencoder_sigmoid_25","autoencoder_sigmoid_2","denoising_autoencoder_sigmoid_2","kitsune"
    # "autoencoder_relu_2" ,"autoencoder_relu_2_filtered_0.2","autoencoder_relu_2_D", "autoencoder_sigmoid_2","autoencoder_sigmoid_2_filtered_0.2","autoencoder_sigmoid_2_D",
    # "denoising_autoencoder_sigmoid_2","denoising_autoencoder_sigmoid_2_filtered_0.2","denoising_autoencoder_sigmoid_2_D"
    ]
    epochs=["1","20","40","60","80","100"]
    
    model_names=[f"{i}_{j}" for i, j in itertools.product(nids_models, epochs)]
    nids_models=[get_nids_model(name,"99.9") for name in model_names]
    planes=[custom_interpolated_plane]
    idx=[custom_point]
    # titles=[f"{i.abbrev} Epoch: {j}" for i,j in itertools.product(nids_models, epochs)]
    titles=None
    #"grad_attrib-ae","grad_attrib-pca","random","Cam_1", "stochastic", [[-5,1,101],[-5,1,101]]
    compare_db_contour(nids_models, planes, idx, "All", feature_range, n_cols=6, titles=titles, plot_range=[[-8,3,201],[-7,5,201]], backend="sns")
    
    
    chars=[]
    nids_models=[
    "autoencoder_relu_2_DR" ,"autoencoder_sigmoid_2_DR","denoising_autoencoder_sigmoid_2_DR"
    # "autoencoder_relu_2" ,"autoencoder_sigmoid_25","autoencoder_sigmoid_2","denoising_autoencoder_sigmoid_2","kitsune"
    ]
    best_epochs=['40',"60",'1',"100","100"]
    
    #"Cam_1", "ACK-LM-0.1-10", "ACK-LM-0.5-10"
    for s in ["All"]:
        iter=itertools.product(nids_models, epochs)
        # iter=zip(nids_models, best_epochs)
        for name,epoch in iter:
            model_name=f"{name}_{epoch}"
            
            # nids_model=get_nids_model(model_name, threshold_key="99.9")
            
            # db_path=f"../adversarial_data/Cam_1/{name}_{nids_model.threshold:.3f}"
            # boundary_folders=[f"{db_path}/{i}"  for i in os.listdir(db_path) if i.startswith("42_Cam_1")]

            # eval_plane(boundary_folders, nids_model, feature_range, eps=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1])

            
            char=plot_plane_metrics(model_name, nids_db[model_name]["thresholds"]["99.9"], s)
            chars.append(char)
            
    chars=pd.concat(chars)
    
    chars=chars.query("plane in ['Interpolated']")
    
    # chars=chars.query("nids != 'Kit' or epoch != 1 ")

    chars=pd.melt(chars, value_vars=["complexity","area"],ignore_index=False)  
    chars=chars.reset_index()
    # print(chars)
    print(chars[["nids","epoch","value"]][(chars["variable"]=="area")& (chars["plane"]=="Interpolated")].set_index(["nids","epoch"]).to_dict())
    # print(chars.to_csv())
    chars.rename(columns={"nids":"NIDS","epoch":"Epoch"},inplace=True)
    fig=sns.relplot(kind="line",data=chars, x="Epoch",y="value", hue="NIDS", row="variable", height=2, aspect=1.5,
                    facet_kws={"sharey":False})
    fig.axes[0,0].set_ylabel("Average Complexity")
    fig.axes[1,0].set_ylabel("Average Area")
    fig.set_titles('')
    
    
    
    # for (row_key, col_key),ax in fig.axes_dict.items():
    #     ax.set_title(f"{row_key}")
    #     ax.set_ylabel(f"{col_key}")
    # fig=sns.catplot(kind="bar", data=chars, x="plane", y="value",col="variable", hue="nids", height=1.5, aspect=2, sharey=False)
    
    fig.tight_layout()
    fig.savefig("exp_figs/meta_plots/summary_plot.pdf")
