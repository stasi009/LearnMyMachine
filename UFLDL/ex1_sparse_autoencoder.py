
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
    n_features = 10
    n_hidden = 2
    n_samples = 10
    X = np.random.uniform(0,255,(n_samples,n_features))

    sae = sparse_autoencoder.SparseAutoEncoder(n_features,2,l2=0,expected_rho=0.1,sparse_beta=0)
    
    weights = sae.weights_vector()

    sae.check_gradients(X,weights)    


