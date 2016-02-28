
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
    n_samples = 200
    n_features = 50
    X = np.random.uniform(0,255,(n_samples,n_features))

    n_hidden = 20
    sae = sparse_autoencoder.SparseAutoEncoder(n_features,n_hidden,l2=3e-3,expected_rho=0.1,sparse_beta=3)
    
    weights = sae.weights_vector()
    epsilon = 1e-5
    sae.check_gradients(X,weights,epsilon)    


