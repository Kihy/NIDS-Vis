from helper import * 
import numpy as np 

class Partitioner:
    def __init__(self, size, scaler):
        self.size=size 
        self.scaler=scaler
        self.edges=np.linspace(0,1,int(1/size)+1)
        self.occupied=set()
        self.features=[]
    
    def process(self, x):
        x_n=self.scaler.transform(x)
        bin_id=map(tuple, np.digitize(x_n, self.edges))
        for idx, feature in zip(bin_id, x):
            if idx not in self.occupied:
                self.occupied.add(idx)
                self.features.append(feature)

    def n_rows(self):
        return len(self.occupied)
    def save_features(self,filename):
        
        with open(filename, "w") as f:
            np.savetxt(f, self.features, delimiter=",")
        

def partition(file, scaler, size):
    batch_size=1024
    traffic_ds = get_dataset(file["path"], batch_size, ndim=46, total_rows=file["total_rows"],
                                scaler=None, frac=1, read_with="tf", dtype="float32",
                                skip_header=True, shuffle=True)
    p=Partitioner(size, scaler)
    
    for data in tqdm(traffic_ds):
        
        p.process(data)
    
    split_file={
        "abbrev": f"{file['name']}_F0.2",
        "path":f"../data/{file['name']}_filtered_{size}.csv",
        "total_rows": p.n_rows(),
        "frac": 1
    }
    
    p.save_features(f"../data/{file['name']}_filtered_{size}.csv")
    
    
    with open("configs/files.json") as f:
        file_db=json.load(f)    
    file_db[f"{file['name']}_filtered_0.2"]=split_file
    with open("configs/files.json","w") as f:
        file_db=json.dump(file_db, f, indent=4)
    
    

def train_split(file, perc, out_file):
    batch_size=1024
    total_rows=int(file["total_rows"]*perc)
    traffic_ds = get_dataset(file["path"], batch_size,ndim=46, total_rows=total_rows,
                                scaler=None, frac=1, read_with="tf", dtype="float32",
                                skip_header=True, shuffle=False)
    
    size=0
    with open(out_file["path"], "w") as f:
        for i in tqdm(traffic_ds):
            np.savetxt(f, i, delimiter=",")
            size+=i.shape[0]
            
    with open("configs/files.json") as f:
        file_db=json.load(f)
    out_file["total_rows"]=size
    file_db[f"{file['name']}_train"]=out_file
    with open("configs/files.json","w") as f:
        file_db=json.dump(file_db, f, indent=4)
    
def convert_sk_scaler(sk_scaler, tf_path):
    
    tf_scaler=MinMaxScaler()
    tf_scaler.data_min_=tf.convert_to_tensor(sk_scaler.data_min_,dtype="float32")
    tf_scaler.data_max_=tf.convert_to_tensor(sk_scaler.data_max_,dtype="float32")
    tf_scaler.first_fit=False
    with open(tf_path, "wb") as f:
        pickle.dump( tf_scaler,f)

if __name__=="__main__":
    dataset="CICIoT"
    scaler_path = f"../../mtd_defence/models/uq/autoencoder/{dataset}_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    convert_sk_scaler(scaler, f"../../mtd_defence/models/uq/autoencoder/{dataset}_min_max_scaler.pkl")
    
    # out_file={"abbrev":"CICIoT_T",
	# 	"path": f"../data/{dataset}_train.csv",
	# 	"total_rows": 0,
	# 	"frac":1}
    
    # files=[dataset]
    # files=get_files(files)
    # train_split(files[0], 0.8, out_file)
    
    
    # files=[f"{dataset}_train"]
    # files=get_files(files)
       
    # partition(files[0], scaler, 0.2)

    
    