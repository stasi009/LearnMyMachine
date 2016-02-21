
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
    def add_bias(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:         raise AttributeError('`how` must be `column` or `row`')
        return X_new

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

    def feedforward(self,X) :
        return self.__feedforward(X,self.W)

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
        result: [n_output,n_sample] matrix
        """
        activation = expit(X)
        return w.dot(Utility.add_bias(activation,how="row"))

    def feedforward(self,X):
        return self.__feedforward(X,self.W)

class OutputBlock(object):
    def __init__(self,n_digits):
        pass

    def feedforward(self,X):
        return expit(X)


