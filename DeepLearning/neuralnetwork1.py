
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
        returned result is a [n_digits,n_samples] matrix
        each row is a digit, each column is a sample
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
        self.velocity = np.zeros_like(self.W)

    def __feedforward(self,X,w):
        """
        X: [n_samples,n_features]
        Xextend: [n_sample,n_features+1]
        w: [n_hidden,n_features+1]
        result: [n_hidden,n_samples] matrix
        """
        self.Xextend = Utility.add_bias(X,how="column")
        return w.dot(self.Xextend.T)

    def feedforward(self,X) :  return self.__feedforward(X,self.W)

    def l2_penalty(self,l2): return Utility.l2_penalty(l2,self.W)

    def backpropagate(self,grad_cost_output,l2):
        # grad_cost_w: [n_hidden,n_features+1] matrix
        # grad_cost_output: [n_hidden,n_sample] matrix
        # Xextend: [n_sample,n_features+1]
        self.grad_cost_w = grad_cost_output.dot(self.Xextend)
        self.grad_cost_w[:,1:] += l2 * self.W[:,1:]

    def update_weights(self,stepsize,shrink_velocity):
        self.velocity = shrink_velocity * self.velocity - stepsize * self.grad_cost_w
        self.W += self.velocity

class HiddenBlock(object):
    def __init__(self,n_hidden,n_output):
        """
        W is [n_output,n_hidden+1] matrix
        """
        self.W = Utility.init_weights(n_hidden,n_output)
        self.velocity = np.zeros_like(self.W)

    def __feedforward(self,X,w):
        """
        X: input, [n_hidden,n_samples] matrix
        w: [n_output,n_hidden+1]
        self.activation: [n_hidden+1,n_sample]
        result: [n_output,n_sample] matrix
        """
        activation = expit(X)
        self.activation = Utility.add_bias(activation,how="row")
        return w.dot(self.activation)

    def feedforward(self,X):    return self.__feedforward(X,self.W)

    def l2_penalty(self,l2): return Utility.l2_penalty(l2,self.W)

    def backpropagate(self,grad_cost_output,l2):
        # grad_cost_ouput: [n_output,n_sample] matrix
        # activation: [n_hidden+1,n_sample]
        # grad_cost_w: [n_output,n_hidden+1] matrix
        self.grad_cost_w = grad_cost_output.dot(self.activation.T)
        self.grad_cost_w[:,1:] += l2 * self.W[:,1:]

        # grad_cost_ouput: [n_output,n_sample] matrix
        # W: [n_output,n_hidden+1] matrix
        # grad_cost_activation: [n_hidden+1,n_sample] matrix
        grad_cost_activation = self.W.T.dot(grad_cost_output)

        # point-wise multiplication, not matrix multiplication
        # three [n_hidden,n_sample] matrix pointwise multiplication, result is
        # also a [n_hidden,n_sample] matrix
        nobias_activation = self.activation[1:,:]
        return grad_cost_activation[1:,:] * nobias_activation * (1 - nobias_activation)

    def update_weights(self,stepsize,shrink_velocity):
        self.velocity = shrink_velocity * self.velocity - stepsize * self.grad_cost_w
        self.W += self.velocity

class OutputBlock(object):
    def __init__(self):
        pass

    def feedforward(self,X):
        """
        X and output: [n_digits,n_sample] matrix
        """
        self.activation = expit(X)
        return self.activation

    def cost(self,Yohe):
        """ todo: this isn't cross-entropy error, need to be revised later """
        term1 = -Yohe * (np.log(self.activation))
        term2 = (1 - Yohe) * np.log(1 - self.activation)
        return np.sum(term1 - term2)

    def backpropagate(self,Yohe):
        """ 
        Yohe: OneHotEncoded Y,[n_digits,n_samples] matrix
        return gradient wrt inputs: [n_digits,n_samples] matrix
        """
        return self.activation - Yohe

class NeuralNetwork(object):

    def __init__(self,n_features,n_hidden,n_digits):
        self.n_digits = n_digits
        self._input = InputBlock(n_features,n_hidden)
        self._hidden = HiddenBlock(n_hidden,n_digits)
        self._output = OutputBlock()

    def predict(self,X):
        """
        X: [n_samples,n_features]
        return: n_sample array
        """
        output_from_input = self._input.feedforward(X)
        # output_from_hidden: [n_digits,n_sample] matrix
        output_from_hidden = self._hidden.feedforward(output_from_input)
        return np.argmax(output_from_hidden,axis=0)

    def __feedforward(self,X,Yohe,l2):
        output_from_input = self._input.feedforward(X)
        # output_from_hidden: [n_digits,n_sample] matrix
        output_from_hidden = self._hidden.feedforward(output_from_input)
        self._output.feedforward(output_from_hidden)
        return self._output.cost(Yohe) + self._input.l2_penalty(l2) + self._hidden.l2_penalty(l2)

    def __backpropagate(self,Yohe,l2):
        grad_output_input = self._output.backpropagate(Yohe) # gradient on output_block's input
        grad_hidden_input = self._hidden.backpropagate(grad_output_input,l2) # gradient on hidden_block's input
        self._input.backpropagate(grad_hidden_input,l2)

    def fit(self,X,y,params):
        l2 = params.get("l2",0.1)
        epochs = params.get("epochs",1000)
        minibatches = params.get("minibatches",50)
        learnrate = params.get("learnrate",0.001)
        shrink_learnrate = params.get("shrink_learnrate",0.00001)
        shuffle = params.get("shuffle",True)

        # Yohe: [n_digits,n_samples]
        Yohe = Utility.encode_labels(y,self.n_digits)

        costs = []
        for index in xrange(epochs):
            # adaptive learning rate
            learnrate /= (1 + shrink_learnrate * index)

            if shuffle:
                idx = np.random.permutation(y.shape[0])
                X,Yohe = X[idx],Yohe[idx]

            mini_indices = np.array_split(range(y.shape[0]), minibatches)
            batch_costs = 0
            for idx in mini_indices:
                Xbatch,Ybatch = X[idx],Yohe[:,idex]

                # ------------------ feed forward
                batch_costs += self.__feedforward(Xbatch,Ybatch,l2)
                
                # ------------------ back propagate
                self.__backpropagate(Ybatch,l2)

                # ------------------ update weights
                self._input.update_weights(learnrate,shrink_velocity)
                self._hidden.update_weights(learnrate,shrink_velocity)

            # cost may oscillate across batches, so average them
            costs.append(batch_costs / minibatches)

        return costs








