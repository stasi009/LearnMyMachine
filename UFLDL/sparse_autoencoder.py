
import numpy as np
import scipy.optimize 
from scipy.special import expit
import commfuncs

class InputBlock(object):
    def __init__(self,n_features,n_hidden,l2):
        # W is a [H,F+1] matrix
        self.W = commfuncs.init_weights(n_features,n_hidden)
        self.l2 = l2

    def _feedforward(self,X,w):
        # X: [S,F]
        # Xextend: [S,F+1]
        # w: [H,F+1]
        # result: [H,S] matrix
        self.Xextend = commfuncs.add_bias(X,how="column")
        return w.dot(self.Xextend.T)

    def feedforward(self,X) :  return self._feedforward(X,self.W)

    def penalty(self): return commfuncs.l2_penalty(self.l2,self.W)

    def backpropagate(self,grad_cost_output):
        # grad_cost_w: [H,F+1] matrix
        # grad_cost_output: [H,S] matrix
        # Xextend: [S,F+1]
        self.grad_cost_w = grad_cost_output.dot(self.Xextend)
        self.grad_cost_w[:,1:] += self.l2 * self.W[:,1:]

class HiddenBlock(object):
    def __init__(self,n_hidden,n_output,l2,expected_rho,sparse_beta):
        """
        expected_rho: expected average activation
        sparse_beta:  weight term for the sparse constraint
        """
        # W is [O,H+1] matrix
        self.W = commfuncs.init_weights(n_hidden,n_output)
        self.l2 = l2
        self.expected_rho = expected_rho

    def _feedforward(self,X,w):
        # X: input, [H,S] matrix
        # self.activation: [H+1,S]
        # w: [O,H+1]
        activation = expit(X)
        
        # rho_hat: actual average activation,[H] vector
        self.rho_hat = np.mean(activation,axis=1)

        # result: [O,S] matrix
        self.activation = commfuncs.add_bias(activation,how="row")
        return w.dot(self.activation)

    def feedforward(self,X):    return self._feedforward(X,self.W)

    def penalty(self): 
        l2penaly = commfuncs.l2_penalty(self.l2,self.W)
        sparse_constraint = self.sparse_beta * commfuncs.KL_divergence(self.expected_rho,self.rho_hat).sum()
        return l2penaly + sparse_constraint

    def backpropagate(self,grad_cost_output):
        num_samples = self.activation.shape[1]

        # grad_cost_ouput: [O,S] matrix
        # activation: [H+1,S]
        # grad_cost_w: [O,H+1] matrix
        self.grad_cost_w = grad_cost_output.dot(self.activation.T)
        self.grad_cost_w[:,1:] += self.l2 * self.W[:,1:]# gradient from l2 regularization

        # grad_cost_ouput: [O,S] matrix
        # W: [O,H+1] matrix
        # grad_cost_activation: [H+1,S] matrix
        grad_activation = self.W.T.dot(grad_cost_output)

        # gradient of "sparse constraints" vs. "activation"
        grad_sparse_activation =  (self.sparse_beta / float(num_samples)) * (-self.expected_rho/self.rho_hat + (1-self.expected_rho)/(1-self.rho_hat)) 
        grad_sparse_activation = np.tile(grad_sparse_activation,(num_samples,1)).T # [H,S] matrix
        grad_activation[1:,:] += grad_sparse_activation

        # point-wise multiplication, not matrix multiplication
        # three [H,S] matrix pointwise multiplication, result is
        # also a [H,S] matrix
        nobias_activation = self.activation[1:,:]
        return grad_activation[1:,:] * nobias_activation * (1 - nobias_activation)

class OutputBlock(object):

    def feedforward(self,X):
        """ X and output: [O,S] matrix """
        self.activation = expit(X)
        return self.activation

    def cost(self,Y):
        """ Y: [O,S] matrix """
        num_samples = Y.shape[1]
        return np.sum((self.activation - Y) ** 2)/(2.0*num_samples)

    def backpropagate(self,Y):
        """ 
        Y: [O,S] matrix
        return gradient wrt inputs: [O,S] matrix
        """
        num_samples = Y.shape[1]
        return (self.activation - Y) * self.activation * (1-self.activation)/(float(num_samples))

class SparseAutoEncoder(object):

    def __init__(self,n_features,n_hidden,**params):
        l2 = params["l2"]
        expected_rho = params["expected_rho"]
        sparse_beta = params["sparse_beta"]

        self._input = InputBlock(n_features,n_hidden,l2=l2)
        self._hidden = HiddenBlock(n_hidden,n_features,l2=l2,expected_rho=expected_rho,sparse_beta=sparse_beta)
        self._output = OutputBlock()

    def __feedforward(self,X,Y):
        output_from_input = self._input.feedforward(X)
        output_from_hidden = self._hidden.feedforward(output_from_input)
        self._output.feedforward(output_from_hidden)
        return self._output.cost(Y) + self._input.penalty() + self._hidden.penalty()

    def __backpropagate(self,Y):
        grad_output_input = self._output.backpropagate(Y) # gradient on output_block's input
        grad_hidden_input = self._hidden.backpropagate(grad_output_input) # gradient on hidden_block's input
        self._input.backpropagate(grad_hidden_input)

    def __assign_weights(self,param_vector):
        offset = 0
        offset,self._input.W = commfuncs.extract_param_matrix(param_vector,offset,self._input.W)
        offset,self._hidden.W = commfuncs.extract_param_matrix(param_vector,offset,self._hidden.W)
        assert offset==len(param_vector)

    def __objective_gradients(self,param_vector):
        self.__assign_weights(param_vector)

        cost = self.__feedforward(self.X,self.X)
        self.__backpropagate(self.X)

        gradients = np.r_[self._input.grad_cost_w.flatten(),self._hidden.grad_cost_w.flatten()]
        return cost,gradients

    def fit(self,X,method="L-BFGS-B",maxiter=400):
        self.X = X

        options = {'maxiter': maxiter, 'disp': True}
        w0 = np.r_[self._input.W.flatten(),self._hidden.W.flatten()]
        result = scipy.optimize.minimize(self.__objective_gradients, w0, method=method, jac=True, options=options)

        self.__assign_weights( result.x )


