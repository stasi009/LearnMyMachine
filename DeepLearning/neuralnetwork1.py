
import numpy as np
from scipy.special import expit

class Utility(object):
    @staticmethod
    def init_weights(num_local_neurons,num_next_neurons):
        nrows = num_next_neurons
        ncols = num_local_neurons + 1# include the bias term
        self.W = np.random.uniform(-1.0, 1.0,size=nrows * ncols).reshape(nrows,ncols)

    @staticmethod
    def encode_labels(y, k):
        """
        returned result is a matrix, each row is a digit, each column is a sample
        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    @staticmethod
    def add_bias(X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:         raise AttributeError('`how` must be `column` or `row`')
        return X_new

    @staticmethod
    def l2_penalty(l2,W):
        """
        W: [num_next_neurons,1+num_local_neurons] matrix
        """
        return 0.5 * l2 * np.sum(W[:,1:] ** 2)# exclude 1st column which is for bias
class InputBlock(object):
    def __init__(self,n_features,n_hidden):
        # W is a [n_hidden,n_features+1] matrix
        self.W = Utility.init_weights(n_features,n_hidden)

    def __feedforward(self,X,w):
        """
        X: [n_samples,n_features]
        w: [n_hidden,n_features+1]
        result: [n_hidden,n_samples] matrix
        """
        X_extend = Utility.add_bias(X,how="column")
        return w.dot(X_extend.T)

    def feedforward(self,X) :  return self.__feedforward(X,self.W)

    def l2_penalty(self,l2): return Utility.l2_penalty(l2,self.W)

    def gradient(self):
        pass
        

class HiddenBlock(object):
    def __init__(self,n_hidden,n_output):
        """
        W is [n_output,n_hidden+1] matrix
        """
        self.W = Utility.init_weights(n_hidden,n_output)

    def __feedforward(self,X,w):
        """
        X: input, [n_hidden,n_samples] matrix
        w: [n_output,n_hidden+1]
        self.activation: [n_hidden+1,n_sample]
        result: [n_output,n_sample] matrix
        """
        activation = expit(X)
        self.activation= Utility.add_bias(activation,how="row")
        return w.dot(self.activation)

    def feedforward(self,X):    return self.__feedforward(X,self.W)

    def l2_penalty(self,l2): return Utility.l2_penalty(l2,self.W)

    def backpropagate(self,grad_cost_output,l2):
        # grad_cost_ouput: [n_output,n_sample] matrix
        # W: [n_output,n_hidden+1] matrix
        # grad_cost_activation: [n_hidden+1,n_sample] matrix
        grad_cost_activation = self.W.T.dot(grad_cost_output)

        # grad_cost_ouput: [n_output,n_sample] matrix
        # activation: [n_hidden+1,n_sample]
        # grad_cost_w: [n_output,n_hidden+1] matrix
        self.grad_cost_w = grad_cost_output.dot(self.activation.T)

        # point-wise multiplication, not matrix multiplication
        # three [n_hidden,n_sample] matrix pointwise multiplication, result is also a [n_hidden,n_sample] matrix
        nobias_activation = self.activation[1:,:]
        return grad_cost_activation[1:,:] * nobias_activation * (1-nobias_activation)

class OutputBlock(object):
    def __init__(self,ytarget):
        """
        ytarget: [n_digits,n_samples] matrix
        """
        self.ytarget = ytarget

    def feedforward(self,X):
        """
        X and output: [n_digits,n_sample] matrix
        """
        self.activation = expit(X)
        return self.activation

    def backpropagate(self):
        """ return gradient wrt inputs: [n_digits,n_samples] matrix"""
        return self.activation - self.ytarget


