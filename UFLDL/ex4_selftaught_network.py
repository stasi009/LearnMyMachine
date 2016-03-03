
import numpy as np
import pandas as pd
import mnistdatas
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sparse_autoencoder import SparseAutoEncoder
from softmax_network import LogisticRegression

class SelfTaughtNetwork(object):

    def __init__(self,n_features,n_hidden,n_output,params):
        self.sae = SparseAutoEncoder(n_features,n_hidden,params["sae_l2"],params["expected_rho"],params["sparse_beta"])
        self.lr = LogisticRegression(n_hidden,n_output,params["lr_l2"])

    def learn_features(self,Xunlabeled,maxiter=400):
        self.sae.fit(Xunlabeled,maxiter=maxiter)
        self.sae.visualize_meta_features()

    def fit(self,X,y,maxiter=400):
        hidden_features = self.sae.extract_features(X,"byrow")
        self.lr.fit(hidden_features,y,maxiter=maxiter)

    def predict(self,X):
        hidden_features = self.sae.extract_features(X,"byrow")
        return self.lr.predict(hidden_features)

############################################ load mnist data
mnist = mnistdatas.MnistDataset("../datas")
mnist.load_train()
mnist.random_plot_different_digits()
mnist.random_plot_same_digits(9)

############################################ split the data
unlabeled = mnist.ytrain >= 5
Xunlabeled = mnist.Xtrain[unlabeled,:]

Xlabeled = mnist.Xtrain[~unlabeled,:]
ylabeled = mnist.ytrain[~unlabeled]

# we have to split training dataset to get test set
# we cannot use provided "mnist test", because the training will only perform on 0~4
Xtrain,Xtest,ytrain,ytest = train_test_split(Xlabeled,ylabeled,test_size=0.4)
pd.value_counts(ytrain)
pd.value_counts(ytest)

############################################ learn hidden features
params = {}
params["sae_l2"] = 3e-3
params["expected_rho"] = 0.1
params["sparse_beta"] = 3
params["lr_l2"] = 1e-4
stl = SelfTaughtNetwork(n_features = mnist.Xtrain.shape[1],n_hidden=196,n_output=5,params=params)

stl.learn_features(Xunlabeled,maxiter=400)

############################################ fit the LR part
stl.fit(Xtrain,ytrain,maxiter=200)

############################################ train accuracy
predicted_ytrain = stl.predict(Xtrain)
print "Train Accuracy: %3.2f%%" % (accuracy_score(ytrain,predicted_ytrain) * 100)
# Train Accuracy: 98.67%

############################################ test accuracy
predicted_ytest = stl.predict(Xtest)
print "Test Accuracy: %3.2f%%" % (accuracy_score(ytest,predicted_ytest) * 100)
# Test Accuracy: 98.37%