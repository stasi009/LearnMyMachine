﻿
import numpy as np

class SimplePCA(object):
    
    def __init__(self,n_components=None,whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.__epsilon = 1e-6

    def fit_transform(self,X, method="svd"):
        """
        X: [n_samples,n_features]
        method="svd", SVD on the orignal/raw matrix, saving the complexity calculating covariance matrix
        method="evd", Eigen-Value Decomposition on covariance matrix, dealing a d*d matrix, use less memory
        """
        feature_mean = X.mean(axis=0)
        # although only "center mean" is necessary, we also need to have unit
        # variance, to ignore the impact caused by different unit
        feature_std = X.std(axis=0)
        self.Xnormalized = (X - feature_mean) / feature_std 

        if method.lower() == "svd":
            U,self.S,Vt = np.linalg.svd(self.Xnormalized)
            self.V = Vt.T
        elif method.lower() == "evd":
            covariance =  self.Xnormalized.T.dot(self.Xnormalized)
            self.V,S, _ = np.linalg.svd(covariance)
            self.S = np.sqrt(S)
        else:
            raise Exception("unknown method")

        if self.n_components is None:
            self.n_components = X.shape[1]# use all features, no reduction

        self.Xprojected = self.Xnormalized.dot(self.V[:,:self.n_components])

        if self.whiten:
            self.Xwhiten = self.Xprojected / (self.S[:self.n_components] + self.__epsilon)
            return self.Xwhiten
        else:
            return self.Xprojected

    def reconstruct(self):
        return self.Xprojected.dot(self.V[:,:self.n_components].T)

