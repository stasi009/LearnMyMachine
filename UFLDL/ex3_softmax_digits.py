
import numpy as np
import mnistdatas
from softmax_network import Network

# ----------------- load mnist data
mnist = mnistdatas.MnistDataset("../datas")
mnist.load_train()
mnist.random_plot_different_digits()
mnist.random_plot_same_digits(9)

def check_gradients():
    n_samples = 30
    n_features = 50
    X = np.random.uniform(0,1,(n_samples,n_features))

    n_hidden = 20
    network = Network(n_features,n_hidden,n_output=10,l2=3e-3)
    
    weights = network.weights_vector()
    network.check_gradients(X,weights)    