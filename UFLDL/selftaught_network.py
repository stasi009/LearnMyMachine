
import numpy as np
import commfuncs
from sparse_autoencoder import SparseAutoEncoder
from softmax_network import SoftmaxRegressor

class SelfTaughtNetwork(object):
    """
    simple Self-Taught-Learning neural network
    only has one SparseAutoEncoder and one Softmax Regressor
    """

    def __init__(self,n_features,n_hidden,n_output,params):
        self.sae = SparseAutoEncoder(n_features,n_hidden,params["sae_l2"],params["expected_rho"],params["sparse_beta"])
        self.lr = SoftmaxRegressor(n_hidden,n_output,params["lr_l2"])
        self._n_output = n_output

    def pretrain_unlabeled(self,Xunlabeled,maxiter=400):
        self.sae.fit(Xunlabeled,maxiter=maxiter)
        self.sae.visualize_meta_features()

    def pretrain_labeled(self,X,y,maxiter=400):
        hidden_features = self.sae.feedforward(X,"byrow")
        self.lr.fit(hidden_features,y,maxiter=maxiter)

    def __assign_weights(self,weights):
        offset = 0
        offset = commfuncs.extract_weights(self.sae._input, weights,offset)
        offset = commfuncs.extract_weights(self.lr._input,weights,offset)
        assert (offset == len(weights))

    def __cost_gradients(self,weights):
        self.__assign_weights(weights)

        # feedforward and backpropagate
        sae_output = self.sae.feedforward(self.X)
        cost,grad_lr_input = self.lr.feedforward_backpropagate(sae_output,self.Yohe)
        self.sae.backpropagate(grad_lr_input)

        return cost,np.r_[self.sae._input.grad_cost_w.flatten(),self.lr._input.grad_cost_w.flatten()]

    def fine_tune(self,X,y,maxiter=400):
        self.X = X
        self.Yohe = commfuncs.encode_digits(y,self._n_output)# Yohe is a [O,S] matrix

        options = {'maxiter': maxiter, 'disp': True}
        weights0 = np.r_[self.sae._input.W.flatten(),self.lr._input.W.flatten()]
        result = scipy.optimize.minimize(self.__cost_gradients, weights0, method="L-BFGS-B", jac=True, options=options)

        self.__assign_weights(result.x)

    def predict(self,X):
        hidden_features = self.sae.feedforward(X,"byrow")
        return self.lr.predict(hidden_features)