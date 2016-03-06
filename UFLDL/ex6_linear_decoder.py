import numpy as np
import mnistdatas
import scipy.io
from sparse_autoencoder import SparseAutoEncoder
import display

# ----------------- load mnist data
mnist = mnistdatas.MnistDataset("../datas")
mnist.load_train()
mnist.random_plot_different_digits()
mnist.random_plot_same_digits(9)

def check_gradients(decoder,minvalue,maxvalue):
    n_samples = 30
    n_features = 50
    X = np.random.uniform(minvalue,maxvalue,(n_samples,n_features))

    n_hidden = 20
    sae = SparseAutoEncoder(n_features,n_hidden,l2=3e-3,expected_rho=0.1,sparse_beta=3,decoder=decoder)
    
    weights = sae.all_weights()
    sae.check_gradients(X,X.T,weights)   
    
check_gradients("sigmoid",0,1)
# !!! the tutorial says, when using linear-decoder, no need to scale the inputs
# !!! actually, the claim is wrong, although in theory, LinearDecoder isn't contrained to [0,1]
# !!! but the hidden block still have sigmoid activation
# !!! large input will easy enter the saturation zone, which has numeric issue
# !!! so still, even using LinearDecoder, we still need, or say, a recommended pratice, to scale the inputs 
check_gradients("linear",0,1) 


patches = scipy.io.loadmat('matlab/stlSampledPatches.mat')['patches']
display.display_color_network(patches[:, 0:100], filename='patches_raw.png')