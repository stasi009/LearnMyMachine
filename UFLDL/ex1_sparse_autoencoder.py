﻿
import numpy as np
import mnistdatas
import sparse_autoencoder

# ----------------- load mnist data
mnist = mnistdatas.MnistDataset("../datas")
mnist.load_train()
mnist.random_plot_different_digits()
mnist.random_plot_same_digits(9)

# ----------------- check gradients
# np.random.seed(99)

def random_input(n_samples,n_features):
    row_indices = np.random.choice(mnist.Xtrain.shape[0],n_samples)
    col_indices = np.random.choice(mnist.Xtrain.shape[1],n_features)
    return mnist.Xtrain[np.ix_(row_indices,col_indices)]

def check_gradients():
    n_samples = 30
    n_features = 50
    X = np.random.uniform(0,1,(n_samples,n_features))
    # X = random_input(n_samples,n_features)

    n_hidden = 20
    sae = sparse_autoencoder.SparseAutoEncoder(n_features,n_hidden,l2=3e-3,expected_rho=0.1,sparse_beta=3)
    
    weights = sae.all_weights()
    sae.check_gradients(X,X.T,weights)    

def fit_display():
    n_samples = 10000
    Xtrain = mnist.Xtrain[:n_samples]

    n_features = mnist.Xtrain.shape[1]
    n_hidden = 196
    sae = sparse_autoencoder.SparseAutoEncoder(n_features,n_hidden,l2=3e-3,expected_rho=0.1,sparse_beta=3)

    sae.fit(Xtrain,maxiter=400)
    sae.visualize_meta_features("SparseAutoEncoder_HiddenFeatures.png")





