import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATASET_DIRECTORY = '../data/CICIoT2023/'

df_sets = [k for k in os.listdir(DATASET_DIRECTORY) if k.endswith('.csv')]


X_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
       'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
       'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
       'ece_flag_number', 'cwr_flag_number', 'ack_count',
       'syn_count', 'fin_count', 'urg_count', 'rst_count', 
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
       'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
       'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
       'Radius', 'Covariance', 'Variance', 'Weight', 
]
y_column = 'label'


# benign_file=open("../data/CICIoT2023/benign.csv","w")
# attack_file=open("../data/CICIoT2023/attack.csv","w")
# for dataset in tqdm(df_sets):
#     data=pd.read_csv(DATASET_DIRECTORY+dataset)
#     data[data[y_column]=="BenignTraffic"].drop(columns=y_column).to_csv(benign_file,header=False,index=False)
#     data[data[y_column]!="BenignTraffic"].drop(columns=y_column).to_csv(attack_file,header=False,index=False)

# scaler=MinMaxScaler()

# dataset=pd.read_csv("../data/CICIoT2023/benign.csv", chunksize=1024)
# count=0
# for chunk in tqdm(dataset):
#     chunk=chunk.to_numpy()
#     print(len(X_columns))
#     scaler.fit(chunk)
#     count+=chunk.shape[0]
# print(count)

# # pickle.dump(scaler, open(f"../../mtd_defence/models/uq/autoencoder/CICIoT_scaler.pkl","wb"))

# dataset=pd.read_csv("../data/CICIoT2023/attack.csv", chunksize=1024)
# count=0
# for chunk in tqdm(dataset):
#     count+=chunk.shape[0]
# print(count)

scaler_type = "min_max"
tf_scaler_path=f"../../mtd_defence/models/uq/autoencoder/CICIoT_{scaler_type}_scaler.pkl"
with open(tf_scaler_path, "rb") as f:
    tf_scaler = pickle.load(f)
    
sk_scaler_path=f"../../mtd_defence/models/uq/autoencoder/CICIoT_scaler.pkl"
with open(sk_scaler_path, "rb") as f:
    sk_scaler = pickle.load(f)
    
dataset=pd.read_csv("../data/CICIoT2023/benign.csv", chunksize=1)
for chunk in tqdm(dataset):
    chunk=chunk.to_numpy()
    tf_chunk=tf_scaler.transform(chunk)
    sk_chunk=sk_scaler.transform(chunk) 
    print(np.count_nonzero(np.isnan(tf_chunk)))  
    print(tf_chunk)
    
    print(np.count_nonzero(np.isnan(sk_chunk)))  
    print(sk_chunk)