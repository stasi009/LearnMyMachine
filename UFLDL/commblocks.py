
import numpy as np
import commfuncs

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
        # grad_cost_output: [H,S] matrix
        # grad_cost_w: [H,F+1] matrix
        # Xextend: [S,F+1]
        self.grad_cost_w = grad_cost_output.dot(self.Xextend)
        self.grad_cost_w[:,1:] += self.l2 * self.W[:,1:]

        grad_xextend = grad_cost_output.T.dot(self.W)# [S,F+1]
        return grad_xextend[:,1:] # [S,F]