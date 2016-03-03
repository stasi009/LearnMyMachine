
import numpy as np

def KL_divergence(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

def init_weights(num_local_neurons,num_next_neurons):
    r = np.sqrt(6.0 / (num_local_neurons + num_next_neurons + 1))
    nrows = num_next_neurons
    ncols = num_local_neurons + 1# include the bias term
    # no need to treat bias specially, initialization of bias won't play a big role
    return np.random.uniform(-r, r,size=(nrows,ncols))

def add_bias(X, how='column'):
    if how == 'column':
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
    elif how == 'row':
        X_new = np.ones((X.shape[0] + 1, X.shape[1]))
        X_new[1:, :] = X
    else:         raise AttributeError('`how` must be `column` or `row`')
    return X_new

def l2_penalty(l2,W):
    """
    W: [num_next_neurons,num_local_neurons+1] matrix
    """
    return 0.5 * l2 * np.sum(W[:,1:] ** 2)# exclude 1st column which is for bias

def extract_weights(block,param_vector,offset):
    next_offset = offset + block.W.size
    block.W = param_vector[offset:next_offset].reshape(block.W.shape)
    return next_offset

def encode_digits(y, k):
    """
    always assume y is from [0,1,...,k-1]
    returned result is a [O,S] matrix, each row is a digit, each column is a sample
    """
    onehot = np.zeros((k, y.shape[0]))
    for idx, val in enumerate(y):
        onehot[val, idx] = 1.0
    return onehot

