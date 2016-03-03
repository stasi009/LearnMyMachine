
import numpy as np
import scipy.optimize 
from scipy.special import expit
import commfuncs
import display

class InputBlock(object):
    def __init__(self,n_features,n_hidden,l2):
        # W is a [H,F+1] matrix
        self.W = commfuncs.init_weights(n_features,n_hidden)
        self.l2 = l2

    def feedforward(self,X):
        # X: [S,F]
        # Xextend: [S,F+1]
        # w: [H,F+1]
        # result: [H,S] matrix
        self.Xextend = commfuncs.add_bias(X,how="column")
        return self.W.dot(self.Xextend.T)

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
        self.sparse_beta = sparse_beta

    def feedforward(self,X):
        # X: input, [H,S] matrix
        # activation: [H,S]
        activation = expit(X)
        
        # rho_hat: actual average activation,[H] vector
        # !!! No need to fix to boundary
        # !!! touching boundary only happens when the input isn't properly normalized/scaled
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

class OutputBlock(object):

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

class SparseAutoEncoder(object):

    def __init__(self,n_features,n_hidden,l2,expected_rho,sparse_beta):
        self._input = InputBlock(n_features,n_hidden,l2=l2)
        self._hidden = HiddenBlock(n_hidden,n_features,l2=l2,expected_rho=expected_rho,sparse_beta=sparse_beta)
        self._output = OutputBlock()

    def __assign_weights(self,weights):
        offset = 0
        offset,self._input.W = commfuncs.extract_param_matrix(weights,offset,self._input.W)
        offset,self._hidden.W = commfuncs.extract_param_matrix(weights,offset,self._hidden.W)
        assert offset == len(weights)

    def __cost(self,weights,X,Y):
        """
        X: [S,F]
        Y: [O,S]
        """
        self.__assign_weights(weights)

        output_from_input = self._input.feedforward(X)
        output_from_hidden = self._hidden.feedforward(output_from_input)
        self._output.feedforward(output_from_hidden)

        return self._output.cost(Y) + self._input.penalty() + self._hidden.penalty()

    def __cost_gradients(self,weights):
        # ------------ feedforward to get cost
        cost = self.__cost(weights,self.X,self.X.T)
        
        # ------------ backpropagate to get gradients
        grad_output_input = self._output.backpropagate(self.X.T) # gradient on output_block's input
        grad_hidden_input = self._hidden.backpropagate(grad_output_input) # gradient on hidden_block's input
        self._input.backpropagate(grad_hidden_input)
        
        return cost,np.r_[self._input.grad_cost_w.flatten(),self._hidden.grad_cost_w.flatten()]

    def weights_vector(self): return np.r_[self._input.W.flatten(),self._hidden.W.flatten()]

    def fit(self,X,method="L-BFGS-B",maxiter=400):
        self.X = X

        options = {'maxiter': maxiter, 'disp': True}
        result = scipy.optimize.minimize(self.__cost_gradients, self.weights_vector(), 
                                         method=method, jac=True, options=options)

        self.__assign_weights(result.x)

    def extract_features(self,X,sample_direction="byrow"):
        # X: [S,F] matrix 
        # input's output: [H,S] matrix
        output_from_input = self._input.feedforward(X)
        activation = expit(output_from_input)

        if sample_direction == "bycolumn":
            return activation # activation: [H,S] matrix
        elif sample_direction == "byrow":
            return activation.T # [S,H] matrix
        else:
            raise Exception("unknown sample direction")

    def __numeric_gradients(self,X, weights, epsilon):
        total = len(weights)
        gradients = np.zeros(total)

        for index in xrange(total):
            old_weight = weights[index]

            weights[index] += epsilon
            cost_plus = self.__cost(weights,X,X.T)

            weights[index] -= 2 * epsilon
            cost_minus = self.__cost(weights,X,X.T)

            gradients[index] = (cost_plus - cost_minus) / (2 * epsilon)
            weights[index] = old_weight
            # print "%d/%d numeric gradients, %3.2f%% completes" % (index + 1,total,(index + 1) * 100.0 / total)

        return gradients

    def check_gradients(self,X, weights, epsilon=1e-4):
        self.X = X
        cost, analytic_gradients = self.__cost_gradients(weights)

        numeric_gradients = self.__numeric_gradients(X,weights,epsilon)
        # print "gradients, 1st column is analytic, 2nd column is numeric: \n%s" % (np.c_[analytic_gradients,numeric_gradients])

        norm_difference = np.linalg.norm(numeric_gradients - analytic_gradients)
        norm_numeric = np.linalg.norm(numeric_gradients)
        norm_analytic = np.linalg.norm(analytic_gradients)
        relative_error = norm_difference / (norm_numeric + norm_analytic)

        if relative_error <= 1e-7:   
            print('    OK:      relative error=%e' % relative_error)
        elif relative_error <= 1e-4:      
            print('*** WARNING: relative error=%e' % relative_error)
        else:                        
            raise Exception('!!! PROBLEM: relative error=%e' % relative_error)

    def visualize_meta_features(self,pic_name=None):
        # W is a [H,F+1] matrix
        meta_features = self._input.W[:,1:].transpose() # [F,H] matrix
        # display_image_patch will treat each column as a single image patch
        display.display_image_patches(meta_features,pic_name)





