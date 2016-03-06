import numpy as np
import mnistdatas
import scipy.io
from sparse_autoencoder import SparseAutoEncoder
from pca import SimplePCA
import display

def check_gradients(decoder,minvalue,maxvalue):
    n_samples = 30
    n_features = 50
    X = np.random.uniform(minvalue,maxvalue,(n_samples,n_features))

    n_hidden = 20
    sae = SparseAutoEncoder(n_features,n_hidden,l2=3e-3,expected_rho=0.1,sparse_beta=3,decoder=decoder)
    
    weights = sae.all_weights()
    sae.check_gradients(X,X.T,weights)   

def test_gradients():
    check_gradients("sigmoid",0,1)
    # !!! the tutorial says, when using linear-decoder, no need to scale the inputs
    # !!! actually, the claim is wrong, although in theory, LinearDecoder isn't contrained to [0,1]
    # !!! but the hidden block still have sigmoid activation
    # !!! large input will easy enter the saturation zone, which has numeric issue
    # !!! so still, even using LinearDecoder, we still need, or say, a recommended pratice, to scale the inputs 
    check_gradients("linear",0,1) 

def plot_patches(imageset,topK,filename=None):
    # imageset is organized by row, but display_color_patches expect image on column
    display.display_color_patches(imageset[:topK].T,filename)

# my API expects each sample occupies a row, not column, so we need 'transpose' here
patches = scipy.io.loadmat('matlab/stlSampledPatches.mat')['patches'].T
plot_patches(patches,100)

pca = SimplePCA(whiten=True)
pca_patches = pca.fit_transform(patches,method="evd")# use eigen-value-decomposition to save memory
plot_patches(pca_patches,100)

n_features = patches.shape[1]
n_hidden = 400
sae = SparseAutoEncoder(n_features,n_hidden,l2=3e-3,expected_rho=0.035,sparse_beta=5,decoder="linear")
sae.fit(patches)
sae.visualize_meta_features(color=True)
