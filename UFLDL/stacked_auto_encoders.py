
import numpy as np
from network_base import NeuralNetworkBase
from sparse_autoencoder import SparseAutoEncoder
from softmax_network import SoftmaxRegressor

class StackedAutoEncoderSoftmaxNetwork(NeuralNetworkBase):

    def __init__(self,num_neurons,params):
        total_layers = len(num_neurons)

        self.sae_l2,expected_rho,sparse_beta = params["sae_l2"],params["expected_rho"],params["sparse_beta"]
        self.saes = [  SparseAutoEncoder(num_neurons[index],num_neurons[index + 1],l2=self.sae_l2,expected_rho=expected_rho,sparse_beta=sparse_beta) for index in xrange(total_layers - 2) ]
        self.softmax = SoftmaxRegressor(num_neurons[total_layers - 2],num_neurons[total_layers - 1],l2=params["softmax_l2"])
        
        self.weighted_blocks = [ sae._input for sae in self.saes]
        self.weighted_blocks.append(self.softmax._input)

        self._n_output = num_neurons[total_layers-1]

    def pretrain(self,X,y,maxiter=400):
        prev_features = X
        for index,sae in enumerate(self.saes):
            print "********** start pretraining %d-th SparseAutoEncoder **********" % (index + 1)
            sae.fit(prev_features,maxiter=maxiter)
            sae.visualize_meta_features("SAE%d-FoundPattern.png" % (index + 1))

            prev_features = sae.feedforward(prev_features)
            print "********** %d-th SparseAutoEncoder finish pretraining **********" % (index + 1)

        print "********** start pretraining final SoftMax **********"
        self.softmax.fit(prev_features,y,maxiter=maxiter)
        print "********** final SoftMax finish pretraining **********"

    def _cost(self,X,Y):
        features = X
        for sae in self.saes:
            features = sae.feedforward(features)
        # penalty only contains the weight decay from softmax, we won't consider weight-decay from each SAE
        # so softmax._cost will be enough
        return self.softmax._cost(features,Y)

    def _gradients(self,Y):
        grad_sae_output = self.softmax._gradients(Y)# gradient on softmax's input, [S,H] matrix

        for index in xrange(len(self.saes)-1,-1,-1):
            sae = self.saes[index]
            grad_sae_output = sae.backpropagate(grad_sae_output)

    def predict_proba(self,X):
        features = X
        for sae in self.saes:
            features = sae.feedforward(features)
        return self.softmax.predict_proba(features)

    def finetune(self,X,y,maxiter=400):
        # during fine tune, no weight decay on SparseAutoEncoder
        for sae in self.saes:
            sae._input.l2 = 0

        Yohe = commfuncs.encode_digits(y,self._n_output)# Yohe is a [O,S] matrix
        self._fit(X,Yohe,maxiter=maxiter)

        # restore
        for sae in self.saes:
            sae._input.l2 = self.sae_l2

    def check_gradients(self, X, Y, weights, epsilon=0.0001):
        # during fine tune, no weight decay on SparseAutoEncoder
        for sae in self.saes: sae._input.l2 = 0

        super(StackedAutoEncoderSoftmaxNetwork, self).check_gradients(X, Y, weights, epsilon)

        # restore
        for sae in self.saes: sae._input.l2 = self.sae_l2
