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
    
    def save_features(self,filename):
        print(len(self.features))
        print(len(self.occupied))
        with open(filename, "w") as f:
            np.savetxt(f, self.features, delimiter=",")
        

def partition(file, scaler, size):
    batch_size=1024
    traffic_ds = get_dataset(file["path"], batch_size, total_rows=file["total_rows"],
                                scaler=None, frac=1, read_with="tf", dtype="float32",
                                skip_header=True, shuffle=True)
    p=Partitioner(size, scaler)
    
    for data in tqdm(traffic_ds):
        
        p.process(data)
    
    p.save_features(f"../data/Cam_1_train_filtered_{size}.csv")

if __name__=="__main__":
    scaler_path = f"../../mtd_defence/models/uq/autoencoder/Cam_1_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    files=["Cam_1_train"]
    files=get_files(files)
    
    partition(files[0], scaler, 0.2)