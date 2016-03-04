
import numpy as np
import scipy.optimize 
from scipy.special import expit
import commfuncs
from commblocks import InputBlock
from network_base import NeuralNetworkBase

class HiddenBlock(object):
    def __init__(self,n_hidden,n_output,l2):
        """
        expected_rho: expected average activation
        sparse_beta:  weight term for the sparse constraint
        """
        # W is [O,H+1] matrix
        self.W = commfuncs.init_weights(n_hidden,n_output)
        self.l2 = l2

    def feedforward(self,X):   
        # X: input, [H,S] matrix
        # activation: [H,S]
        activation = expit(X)
        
        # self.activation: [H+1,S]
        # w: [O,H+1]
        # result: [O,S] matrix
        self.activation = commfuncs.add_bias(activation,how="row")
        return self.W.dot(self.activation)

    def penalty(self): 
        return commfuncs.l2_penalty(self.l2,self.W)

    def backpropagate(self,grad_cost_output):
        # grad_cost_ouput: [O,S] matrix
        # activation: [H+1,S]
        # grad_cost_w: [O,H+1] matrix
        self.grad_cost_w = grad_cost_output.dot(self.activation.T)
        self.grad_cost_w[:,1:] += self.l2 * self.W[:,1:]# gradient from l2 regularization

        # grad_cost_ouput: [O,S] matrix
        # W: [O,H+1] matrix
        # grad_cost_activation: [H+1,S] matrix
        grad_activation = self.W.T.dot(grad_cost_output)

        # point-wise multiplication, not matrix multiplication
        # three [H,S] matrix pointwise multiplication,
        # result is also a [H,S] matrix
        nobias_activation = self.activation[1:,:]
        return grad_activation[1:,:] * nobias_activation * (1 - nobias_activation)

class OutputBlock(object):

    def feedforward(self,X):
        # remove the largest, to avoid overflow
        # it is only a trick to avoid numeric issue
        # this trick won't change "activation" and its gradient
        X -= np.max(X)
        """ X and activation: [O,S] matrix """
        Xexp = np.exp(X) # [O,S] matrix
        self.activation = Xexp / (Xexp.sum(axis=0))
        return self.activation

    def cost(self,Y):
        """ Y and activation: [O,S] matrix """
        num_samples = Y.shape[1]
        return -1.0 * np.sum((np.log(self.activation) * Y)) / num_samples

    def backpropagate(self,Y):
        """ 
        Y: [O,S] matrix
        return gradient wrt inputs: [O,S] matrix
        """
        num_samples = Y.shape[1]
        return (self.activation - Y) / (float(num_samples))

class SoftmaxRegressor(NeuralNetworkBase):
    """
    ignore the hidden layer, no nonlinear feature mapping, just a Multi-class Logistic Regression
    """
    def __init__(self,n_features,n_output,l2):
        self._input = InputBlock(n_features,n_output,l2=l2)
        self._output = OutputBlock()
        self._n_output = n_output
        self.weighted_blocks = [self._input]

    def predict_proba(self,X):
        output_from_input = self._input.feedforward(X)
        probas = self._output.feedforward(output_from_input) # [O,S] matrix
        return probas.T # [S,O] matrix

    def _cost(self,X,Y):
        # ------------ feedforward to get cost
        self.predict_proba(X)
        return self._output.cost(Y) + self._input.penalty() 

    def _gradients(self,Yohe):
        # ------------ backpropagate to get gradients
        grad_output_input = self._output.backpropagate(Yohe) # gradient on output_block's input
        grad_input_input = self._input.backpropagate(grad_output_input) # gradient on input block's input
        return grad_input_input # [S,F] matrix

    def fit(self,X,y,method="L-BFGS-B",maxiter=400):
        Yohe = commfuncs.encode_digits(y,self._n_output)
        return self._fit(X,Yohe,method=method,maxiter=maxiter)

class NeuralNetwork(NeuralNetworkBase):

    def __init__(self,n_features,n_hidden,n_output,l2):
        self._input = InputBlock(n_features,n_hidden,l2=l2)
        self._hidden = HiddenBlock(n_hidden,n_output,l2=l2)
        self._output = OutputBlock()
        self.weighted_blocks = [self._input,self._hidden]
        self._n_output = n_output

    def _cost(self,X,Y):
        """
        X: [S,F]
        Y: [O,S]
        """
        output_from_input = self._input.feedforward(X)
        output_from_hidden = self._hidden.feedforward(output_from_input)
        self._output.feedforward(output_from_hidden)
        return self._output.cost(Y) + self._input.penalty() + self._hidden.penalty()

    def _gradients(self,Y):
        grad_output_input = self._output.backpropagate(Y) # gradient on output_block's input
        grad_hidden_input = self._hidden.backpropagate(grad_output_input) # gradient on hidden_block's input
        self._input.backpropagate(grad_hidden_input)

    def predict_proba(self,X):
        output_from_input = self._input.feedforward(X)
        output_from_hidden = self._hidden.feedforward(output_from_input)
        probas = self._output.feedforward(output_from_hidden) # [O,S] matrix
        return probas.T # [S,O] matrix

    def fit(self,X,y,method="L-BFGS-B",maxiter=400):
        Yohe = commfuncs.encode_digits(y,self._n_output)
        return self._fit(X,Yohe,method=method,maxiter=maxiter)

    

    








