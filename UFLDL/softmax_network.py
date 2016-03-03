
import numpy as np
import scipy.optimize 
from scipy.special import expit
import commfuncs
from commblocks import InputBlock

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
        return (self.activation - Y)  / (float(num_samples))

class LogisticRegression(object):
    """
    ignore the hidden layer, no nonlinear feature mapping, just a Multi-class Logistic Regression
    """

    def __init__(self,n_features,n_output,l2):
        self._input = InputBlock(n_features,n_output,l2=l2)
        self._output = OutputBlock()
        self._n_output = n_output

    def predict_proba(self,X):
        output_from_input = self._input.feedforward(X)
        probas = self._output.feedforward(output_from_input) # [O,S] matrix
        return probas.T # [S,O] matrix

    def predict(self,X):
        predicted_probas = self.predict_proba(X)# [S,O] matrix
        return predicted_probas.argmax(axis=1)

    def __cost_gradients(self,weights):
        self._input.W = weights.reshape(self._input.W.shape)

        # ------------ feedforward to get cost
        output_from_input = self._input.feedforward(self.X)
        self._output.feedforward(output_from_input)
        cost = self._output.cost(self.Yohe) + self._input.penalty() 
        
        # ------------ backpropagate to get gradients
        grad_output_input = self._output.backpropagate(self.Yohe) # gradient on output_block's input
        self._input.backpropagate(grad_output_input)
        
        return cost,self._input.grad_cost_w.flatten()

    def fit(self,X,y,method="L-BFGS-B",maxiter=400):
        self.X = X
        self.Yohe = commfuncs.encode_digits(y,self._n_output)# Yohe is a [O,S] matrix

        options = {'maxiter': maxiter, 'disp': True}
        result = scipy.optimize.minimize(self.__cost_gradients, self._input.W.flatten(), method=method, jac=True, options=options)

        self._input.W = result.x.reshape(self._input.W.shape)

class NeuralNetwork(object):

    def __init__(self,n_features,n_hidden,n_output,l2):
        self._input = InputBlock(n_features,n_hidden,l2=l2)
        self._hidden = HiddenBlock(n_hidden,n_output,l2=l2)
        self._output = OutputBlock()
        self._n_output = n_output

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
        cost = self.__cost(weights,self.X,self.Yohe)
        
        # ------------ backpropagate to get gradients
        grad_output_input = self._output.backpropagate(self.Yohe) # gradient on output_block's input
        grad_hidden_input = self._hidden.backpropagate(grad_output_input) # gradient on hidden_block's input
        self._input.backpropagate(grad_hidden_input)
        
        return cost,np.r_[self._input.grad_cost_w.flatten(),self._hidden.grad_cost_w.flatten()]

    def weights_vector(self): return np.r_[self._input.W.flatten(),self._hidden.W.flatten()]

    def fit(self,X,y,method="L-BFGS-B",maxiter=400):
        self.X = X
        self.Yohe = commfuncs.encode_digits(y,self._n_output)# Yohe is a [O,S] matrix

        options = {'maxiter': maxiter, 'disp': True}
        result = scipy.optimize.minimize(self.__cost_gradients, self.weights_vector(), method=method, jac=True, options=options)

        self.__assign_weights(result.x)

    def predict_proba(self,X):
        output_from_input = self._input.feedforward(X)
        output_from_hidden = self._hidden.feedforward(output_from_input)
        probas = self._output.feedforward(output_from_hidden) # [O,S] matrix
        return probas.T # [S,O] matrix

    def predict(self,X):
        predicted_probas = self.predict_proba(X)# [S,O] matrix
        return predicted_probas.argmax(axis=1)

    def __numeric_gradients(self,X,Yohe, weights, epsilon):
        total = len(weights)
        gradients = np.zeros(total)

        for index in xrange(total):
            old_weight = weights[index]

            weights[index] += epsilon
            cost_plus = self.__cost(weights,X,Yohe)

            weights[index] -= 2 * epsilon
            cost_minus = self.__cost(weights,X,Yohe)

            gradients[index] = (cost_plus - cost_minus) / (2 * epsilon)
            weights[index] = old_weight
            # print "%d/%d numeric gradients, %3.2f%% completes" % (index + 1,total,(index + 1) * 100.0 / total)

        return gradients

    def check_gradients(self,X,y, weights, epsilon=1e-4):
        self.X = X
        self.Yohe = commfuncs.encode_digits(y,self._n_output)# Yohe is a [O,S] matrix
        cost, analytic_gradients = self.__cost_gradients(weights)

        numeric_gradients = self.__numeric_gradients(X,self.Yohe,weights,epsilon)
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

    








