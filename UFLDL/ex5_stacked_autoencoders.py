
import numpy as np
import mnistdatas
from sklearn.metrics import accuracy_score
from stacked_auto_encoders import StackedAutoEncoderSoftmaxNetwork
import commfuncs

# ----------------- load mnist data
mnist = mnistdatas.MnistDataset("../datas")
mnist.load_train()
mnist.load_test()
mnist.random_plot_different_digits()
mnist.random_plot_same_digits(9)

# ----------------- configurations
params = {}
params["sae_l2"] = 3e-3
params["expected_rho"] = 0.1
params["sparse_beta"] = 3
params["softmax_l2"] = 1e-4

def check_gradients():
    n_samples = 30
    n_features = 50
    n_output = 10
    X = np.random.uniform(0,1,(n_samples,n_features))
    y = np.random.choice(n_output,n_samples)

    num_neurons = [n_features,20,10,n_output]
    network = StackedAutoEncoderSoftmaxNetwork(num_neurons,params)
    
    weights = network.all_weights()
    Yohe = commfuncs.encode_digits(y,n_output)
    network.check_gradients(X,Yohe,weights)   

def check_accuracy(sae_softmax):   
    # ------ train accuracy
    predicted_ytrain = sae_softmax.predict(mnist.Xtrain)
    print "Train Accuracy: %3.2f%%" % (accuracy_score(mnist.ytrain,predicted_ytrain) * 100)

    # ------ test accuracy
    predicted_ytest = sae_softmax.predict(mnist.Xtest)
    print "Test Accuracy: %3.2f%%" % (accuracy_score(mnist.ytest,predicted_ytest) * 100)

def pretain_finetune():
    n_features = mnist.Xtrain.shape[1]
    n_output = 10
    num_neurons = [n_features,196,196,n_output]
    sae_softmax = StackedAutoEncoderSoftmaxNetwork(num_neurons,params=params)

    # ------ pre-training
    sae_softmax.pretrain(mnist.Xtrain,mnist.ytrain,maxiter=400)

    # Train Accuracy: 91.03%
    # Test Accuracy: 91.72%
    check_accuracy(sae_softmax)

    # ------ fine tune
    sae_softmax.finetune(mnist.Xtrain,mnist.ytrain,maxiter=400)

    # Train Accuracy: 100%
    # Test Accuracy: 98.03%
    check_accuracy(sae_softmax)