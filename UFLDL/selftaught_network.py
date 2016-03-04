
import numpy as np
import commfuncs
from sparse_autoencoder import SparseAutoEncoder
from softmax_network import SoftmaxRegressor
from network_base import NeuralNetworkBase

class SelfTaughtNetwork(NeuralNetworkBase):
    """
    simple Self-Taught-Learning neural network
    only has one SparseAutoEncoder and one Softmax Regressor
    """

    def __init__(self,n_features,n_hidden,n_output,params):
        self.sae = SparseAutoEncoder(n_features,n_hidden,params["sae_l2"],params["expected_rho"],params["sparse_beta"])
        self.softmax = SoftmaxRegressor(n_hidden,n_output,params["softmax_l2"])
        self.weighted_blocks = [self.sae._input,self.softmax._input]
        self._n_output = n_output

    def pretrain_unlabeled(self,Xunlabeled,maxiter=400):
        self.sae.fit(Xunlabeled,maxiter=maxiter)
        self.sae.visualize_meta_features()

    def pretrain_labeled(self,X,y,maxiter=400):
        hidden_features = self.sae.feedforward(X)
        self.softmax.fit(hidden_features,y,maxiter=maxiter)

    def predict_proba(self,X):
        hidden_features = self.sae.feedforward(X)
        return self.softmax.predict_proba(hidden_features)

    def _cost(self,X,Yohe):
        sae_output = self.sae.feedforward(X)# [S,H] matrix
        # penalty only contains the weight decay from softmax, so softmax._cost will be enough
        return self.softmax._cost(sae_output,Yohe)

    def _gradients(self,Y):
        grad_softmax_input = self.softmax._gradients(Y)# [S,H] matrix
        self.sae.backpropagate(grad_softmax_input) # return [S,F] matrix

    def fine_tune(self,X,y,maxiter=400):
        old_sae_l2 = self.sae._input.l2
        self.sae._input.l2 = 0 # during fine tune, no weight decay on SparseAutoEncoder

        Yohe = commfuncs.encode_digits(y,self._n_output)# Yohe is a [O,S] matrix
        self._fit(X,Yohe,maxiter=maxiter)

        self.sae._input.l2 = old_sae_l2 # restore

    def check_gradients(self, X, Y, weights, epsilon = 0.0001):
        old_sae_l2 = self.sae._input.l2
        self.sae._input.l2 = 0 # during fine tune, no weight decay on SparseAutoEncoder

        super(SelfTaughtNetwork, self).check_gradients(X, Y, weights, epsilon)

        self.sae._input.l2 = old_sae_l2 # restore
