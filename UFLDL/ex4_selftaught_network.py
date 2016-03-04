
import numpy as np
import pandas as pd
import mnistdatas
import commfuncs
from selftaught_network import SelfTaughtNetwork
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

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

############################################ configure
params = {}
params["sae_l2"] = 3e-3
params["expected_rho"] = 0.1
params["sparse_beta"] = 3
params["softmax_l2"] = 1e-4

def check_gradients():
    n_samples = 30
    n_features = 50
    n_output = 10
    X = np.random.uniform(0,1,(n_samples,n_features))
    y = np.random.choice(n_output,n_samples)

    n_hidden = 20
    stl = SelfTaughtNetwork(n_features,n_hidden,n_output,params)
    
    weights = stl.all_weights()
    Yohe = commfuncs.encode_digits(y,n_output)
    stl.check_gradients(X,Yohe,weights)
    
def check_accuracy(stl):   
    # ------ train accuracy
    predicted_ytrain = stl.predict(Xtrain)
    print "Train Accuracy: %3.2f%%" % (accuracy_score(ytrain,predicted_ytrain) * 100)

    # ------ test accuracy
    predicted_ytest = stl.predict(Xtest)
    print "Test Accuracy: %3.2f%%" % (accuracy_score(ytest,predicted_ytest) * 100)

def pretain_finetune():
    stl = SelfTaughtNetwork(n_features = mnist.Xtrain.shape[1],n_hidden=196,n_output=5,params=params)

    # ------ pre-training
    stl.pretrain_unlabeled(Xunlabeled,maxiter=400)
    # although set maxiter=200, but since we have high-level features, it only loop 76 times and converge and stop
    stl.pretrain_labeled(Xtrain,ytrain,maxiter=200)

    # Train Accuracy: 98.63%
    # Test Accuracy: 98.41%
    check_accuracy(stl)

    # ------ fine tune
    stl.fine_tune(Xtrain,ytrain)

    # Train Accuracy: 100%
    # Test Accuracy: 99.02%
    check_accuracy(stl)