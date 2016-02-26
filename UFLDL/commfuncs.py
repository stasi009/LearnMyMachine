
import numpy as np

def KL_divergence(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

def init_weights(num_in,num_next_neurons):
    nrows = num_next_neurons
    ncols = num_local_neurons + 1# include the bias term
    return np.random.uniform(-1.0, 1.0,size=nrows * ncols).reshape(nrows,ncols)
