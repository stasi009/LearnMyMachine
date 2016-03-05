
import numpy as np
import mnistdatas
from sklearn.metrics import accuracy_score
from stacked_auto_encoders import StackedAutoEncoderSoftmaxNetwork
import commfuncs

# ----------------- load mnist data
mnist = mnistdatas.MnistDataset("../datas")
mnist.load_train()
mnist.load_test()
mnist.random_plot_different_digits()
mnist.random_plot_same_digits(9)

# ----------------- configurations
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

    num_neurons = [n_features,20,10,n_output]
    network = StackedAutoEncoderSoftmaxNetwork(num_neurons,params)
    
    weights = network.all_weights()
    Yohe = commfuncs.encode_digits(y,n_output)
    network.check_gradients(X,Yohe,weights)   