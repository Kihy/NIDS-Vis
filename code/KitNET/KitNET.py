import numpy as np
import KitNET.dA as AE
import KitNET.corClust as CC
from tqdm import tqdm
# This class represents a KitNET machine learner.
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
# For licensing information, see the end of this document


class KitNET:
    # n: the number of features in your input dataset (i.e., x \in R^n)
    # m: the maximum size of any autoencoder in the ensemble layer
    # AD_grace_period: the number of instances the network will learn from before producing anomaly scores
    # FM_grace_period: the number of instances which will be taken to learn the feature mapping. If 'None', then FM_grace_period=AM_grace_period
    # learning_rate: the default stochastic gradient descent learning rate for all autoencoders in the KitNET instance.
    # hidden_ratio: the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
    # feature_map: One may optionally provide a feature map instead of learning one. The map must be a list,
    #           where the i-th entry contains a list of the feature indices to be assingned to the i-th autoencoder in the ensemble.
    #           For example, [[2,5,3],[4,0,1],[6,7]]
    def __init__(self, n, max_autoencoder_size=10, FM_grace_period=None, AD_grace_period=10000, learning_rate=0.1, hidden_ratio=0.75, feature_map=None, normalize=True, input_precision=None, quantize=None):
        # Parameters:
        self.AD_grace_period = AD_grace_period
        if FM_grace_period is None:
            self.FM_grace_period = AD_grace_period
        else:
            self.FM_grace_period = FM_grace_period
        self.input_precision = input_precision
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n
        self.normalize = normalize
        # Variables
        self.n_trained = 0  # the number of training instances so far
        self.n_executed = 0  # the number of executed instances so far
        self.v = feature_map
        self.ensembleLayer = []
        self.outputLayer = None
        self.quantize = quantize
        self.norm_max = np.ones((n,)) * -np.Inf
        self.norm_min = np.ones((n,)) * np.Inf
        if self.v is None:
            pass
            # print("Feature-Mapper: train-mode, Anomaly-Detector: off-mode")
        else:
            self.__createAD__()
            # print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        # incremental feature cluatering for the feature mapping process
        self.FM = CC.corClust(self.n)

    # If FM_grace_period+AM_grace_period has passed, then this function executes KitNET on x. Otherwise, this function learns from x.
    # x: a numpy array of length n
    # Note: KitNET automatically performs 0-1 normalization on all attributes.
    def process(self, x):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        non_nan_idx = np.where(~np.isnan(x).all(axis=1))

        ret_array = np.zeros(x.shape[0])
        ret_array.fill(np.nan)

        ret_array[non_nan_idx] = self.execute(x[non_nan_idx])
        # if x.shape[0] > 2:
        # print("x", x[non_nan_idx][:2])
        # print("ret", ret_array[non_nan_idx][:2])
        # raise
        return ret_array

    # alias so it is compatible with sklearn models, input and output are all 2d arrays
    def decision_function(self, x):
        anom_score = self.predict(x)
        return -anom_score

    def score_samples(self, x):
        return self.decision_function(x)

    # alias for execute for it is compatible with tf models, processes in batches
    def predict(self, x):
        return self.process(x)
        # return np.array([self.process(x[i]) for i in range(np.array(x).shape[0])])

    # force train KitNET on x
    # returns the anomaly score of x during training (do not use for alerting)
    def train(self, x):
        # If the FM is in train-mode, and the user has not supplied a feature mapping
        if self.n_trained <= self.FM_grace_period and self.v is None:
            # update the incremetnal correlation matrix
            self.FM.update(x)
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
            if self.n_trained == self.FM_grace_period:  # If the feature mapping should be instantiated
                self.v = self.FM.cluster(self.m)
                self.__createAD__()
                # print("The Feature-Mapper found a mapping: "+str(self.n)+" features to "+str(len(self.v))+" autoencoders.")
                # print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        else:  # train
            # Ensemble Layer
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                # make sub instance for autoencoder 'a'
                xi = x[self.v[a]]
                S_l1[a] = self.ensembleLayer[a].train(xi)
            # OutputLayer
            self.outputLayer.train(S_l1)
            if self.n_trained == self.AD_grace_period + self.FM_grace_period:
                pass
                # print("Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode")
        self.n_trained += 1
    def summary(self):
        
        for idx, i in enumerate(self.ensembleLayer):
            print(idx, i.get_num_params())
        print("output", self.outputLayer.get_num_params())
    # force execute KitNET on x
    def execute(self, x):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        if self.v is None:
            raise RuntimeError(
                'KitNET Cannot execute x, because a feature mapping has not yet been learned or provided. Try running process(x) instead.')
        else:
            self.n_executed += x.shape[0]
            # Ensemble Layer
            S_l1 = np.zeros((x.shape[0], len(self.ensembleLayer)))
            for a in range(len(self.ensembleLayer)):
                # make sub inst
                xi = x[:, self.v[a]]

                S_l1[:, a] = self.ensembleLayer[a].execute(xi)
            
            # OutputLayer
            return self.outputLayer.execute(S_l1)

    def __createAD__(self):
        # construct ensemble layer
        for map in self.v:
            params = AE.dA_params(n_visible=len(map), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0,
                                  hiddenRatio=self.hr, normalize=self.normalize, input_precision=self.input_precision, quantize=self.quantize,
                                  norm_max=self.norm_max[map], norm_min=self.norm_min[map])
            self.ensembleLayer.append(AE.dA(params))

        # construct output layer
        params = AE.dA_params(len(self.v), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0, hiddenRatio=self.hr,
                              normalize=self.normalize, quantize=self.quantize, input_precision=self.input_precision)
        self.outputLayer = AE.dA(params)

    def get_params(self):
        return_dict = {"ensemble": []}
        for i in range(len(self.ensembleLayer)):
            return_dict["ensemble"].append(self.ensembleLayer[i].get_params())
        return_dict["output"] = self.outputLayer.get_params()
        return return_dict

    def set_params(self, new_param):
        for i in range(len(new_param["ensemble"])):
            self.ensembleLayer[i].set_params(new_param["ensemble"][i])
        self.outputLayer.set_params(new_param["output"])


# Copyright (c) 2017 Yisroel Mirsky
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
