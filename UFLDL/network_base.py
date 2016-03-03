
import numpy as np
import scipy.optimize 

class NeuralNetworkBase(object):

    def _cost(self,weights,X,Y):
        raise NotImplementedError("to be overriden in derived class")

    def predict_proba(self,X):
        raise NotImplementedError("to be overriden in derived class")

    def _cost_gradients(self,weights):
        raise NotImplementedError("to be overriden in derived class")

    def _assign_weights(self,weights):
        raise NotImplementedError("to be overriden in derived class")

    def weights_vector(self): 
        raise NotImplementedError("to be overriden in derived class")

    def _numeric_gradients(self,X,Y, weights, epsilon):
        total = len(weights)
        gradients = np.zeros(total)

        for index in xrange(total):
            old_weight = weights[index]

            weights[index] += epsilon
            cost_plus = self._cost(weights,X,Y)

            weights[index] -= 2 * epsilon
            cost_minus = self._cost(weights,X,Y)

            gradients[index] = (cost_plus - cost_minus) / (2 * epsilon)
            weights[index] = old_weight
            # print "%d/%d numeric gradients, %3.2f%% completes" % (index +
            # 1,total,(index + 1) * 100.0 / total)

        return gradients

    def check_gradients(self,X,Y, weights, epsilon=1e-4):
        """
        X is a [S,F] matrix
        Y is a [O,S] matrix
        """
        self.X = X
        self.Y = Y 
        cost, analytic_gradients = self._cost_gradients(weights)

        numeric_gradients = self._numeric_gradients(X,Y,weights,epsilon)
        # print "gradients, 1st column is analytic, 2nd column is numeric:
        # \n%s" % (np.c_[analytic_gradients,numeric_gradients])

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

    def predict(self,X):
        predicted_probas = self.predict_proba(X)# [S,O] matrix
        return predicted_probas.argmax(axis=1)

    def fit(self,X,Y,method="L-BFGS-B",maxiter=400):
        self.X = X
        self.Y = Y # Y must be a [O,S] matrix

        options = {'maxiter': maxiter, 'disp': True}
        result = scipy.optimize.minimize(self._cost_gradients, self.weights_vector(), method=method, jac=True, options=options)

        self._assign_weights(result.x)

