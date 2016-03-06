
import numpy as np
from scipy.special import expit
import commfuncs
from commblocks import InputBlock
from network_base import NeuralNetworkBase
import display

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
        self.sparse_beta = sparse_beta

    def feedforward(self,X):
        # X: input, [H,S] matrix
        # activation: [H,S]
        activation = expit(X)
        
        # rho_hat: actual average activation,[H] vector
        # !!!  No need to fix to boundary
        # !!!  touching boundary only happens when the input isn't properly
        # normalized/scaled
        self.rho_hat = np.mean(activation,axis=1)

        # self.activation: [H+1,S]
        # w: [O,H+1]
        # result: [O,S] matrix
        self.activation = commfuncs.add_bias(activation,how="row")
        return self.W.dot(self.activation)

    def penalty(self): 
        l2penaly = commfuncs.l2_penalty(self.l2,self.W)
        sparse_constraint = self.sparse_beta * (commfuncs.KL_divergence(self.expected_rho,self.rho_hat).sum())
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

        # gradient of "sparse constraints" vs.  "activation"
        grad_sparse_activation = (self.sparse_beta / float(num_samples)) * (-self.expected_rho / self.rho_hat + (1 - self.expected_rho) / (1 - self.rho_hat)) 
        grad_sparse_activation = np.tile(grad_sparse_activation,(num_samples,1)).T # [H,S] matrix
        grad_activation[1:,:] += grad_sparse_activation

        # point-wise multiplication, not matrix multiplication
        # three [H,S] matrix pointwise multiplication,
        # result is also a [H,S] matrix
        nobias_activation = self.activation[1:,:]
        return grad_activation[1:,:] * nobias_activation * (1 - nobias_activation)

class SigmoidDecoder(object):

    def feedforward(self,X):
        """ X and output: [O,S] matrix """
        self.activation = expit(X)
        return self.activation

    def cost(self,Y):
        """ Y: [O,S] matrix """
        num_samples = Y.shape[1]
        return np.sum((self.activation - Y) ** 2) / (2.0 * num_samples)

    def backpropagate(self,Y):
        """ 
        Y: [O,S] matrix
        return gradient wrt inputs: [O,S] matrix
        """
        num_samples = Y.shape[1]
        return (self.activation - Y) * self.activation * (1 - self.activation) / (float(num_samples))

class LinearDecoder(object):

    def feedforward(self,X):
        """ X=activation: [O,S] matrix """
        self.activation = X
        return self.activation

    def cost(self,Y):
        """ Y: [O,S] matrix """
        num_samples = Y.shape[1]
        return np.sum((self.activation - Y) ** 2) / (2.0 * num_samples)

    def backpropagate(self,Y):
        """ 
        Y: [O,S] matrix
        return gradient wrt inputs: [O,S] matrix
        """
        num_samples = Y.shape[1]
        return (self.activation - Y)  / (float(num_samples))

class SparseAutoEncoder(NeuralNetworkBase):

    def __init__(self,n_features,n_hidden,l2,expected_rho,sparse_beta,decoder="sigmoid"):
        self._input = InputBlock(n_features,n_hidden,l2=l2)
        self._hidden = HiddenBlock(n_hidden,n_features,l2=l2,expected_rho=expected_rho,sparse_beta=sparse_beta)
        self.weighted_blocks = [self._input,self._hidden]

        if decoder.lower() == "sigmoid":
            self._output = SigmoidDecoder()
        elif decoder.lower() == "linear":
            self._output = LinearDecoder()
        else:
            raise Exception("unsupported decoder type")

    def _cost(self,X,Y):
        output_from_input = self._input.feedforward(X)
        output_from_hidden = self._hidden.feedforward(output_from_input)
        self._output.feedforward(output_from_hidden)
        return self._output.cost(Y) + self._input.penalty() + self._hidden.penalty()

    def _gradients(self,Y):
        grad_output_input = self._output.backpropagate(self.X.T) # gradient on output_block's input
        grad_hidden_input = self._hidden.backpropagate(grad_output_input) # gradient on hidden_block's input
        self._input.backpropagate(grad_hidden_input)

    # since Python doesn't support overload, I have to provide a convenient
    # method to simplify the API
    def fit(self,X,method="L-BFGS-B",maxiter=400):
        self._fit(X,X.T,method=method,maxiter=maxiter)

    def feedforward(self,X):
        """ regard the SparseAutoEncoder as a single block """
        # X: [S,F] matrix
        # input's output: [H,S] matrix
        output_from_input = self._input.feedforward(X)
        self.activation = expit(output_from_input).T # activation: [S,H] matrix
        return self.activation

    def backpropagate(self,grad_hidden_output):
        # grad_hidden_output is gradient w.r.t hidden's output, [S,H] matrix
        # self.activation: [S,H] matrix
        # grad_input_output is gradients w.r.t input's output, [S,H] matrix
        grad_input_output = grad_hidden_output * self.activation * (1 - self.activation)# [S,H] matrix
        # backpropagate expects [H,S] matrix matrix, so transpose
        return self._input.backpropagate(grad_input_output.T)# return [S,F] matrix

    def visualize_meta_features(self,pic_name=None):
        # W is a [H,F+1] matrix
        meta_features = self._input.W[:,1:].transpose() # [F,H] matrix
        # display_image_patch will treat each column as a single image patch
        display.display_patches(meta_features,pic_name)





