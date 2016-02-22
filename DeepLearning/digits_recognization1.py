
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use("ggplot")
from sklearn.metrics import accuracy_score

import commfuncs
import mnistdatas
import neuralnetwork1

mnist = mnistdatas.MnistDataset("../datas")
mnist.load_train()
mnist.random_plot_different_digits()
mnist.random_plot_same_digits(9)

network = neuralnetwork1.NeuralNetwork(n_features = mnist.Xtrain.shape[0],n_hidden=50,n_digits=10)

params = {}
params["l2"] = 0.1
params["epochs"] = 1000
params["minibatches"] = 50
params["learnrate"] = 0.001
params["shrink_learnrate"] = 0.00001
params["shuffle"] = True

costs = network.fit(mnist.Xtrain,mnist.ytrain,params)
