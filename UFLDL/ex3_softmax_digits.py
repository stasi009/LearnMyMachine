
import cPickle
import numpy as np
import mnistdatas
from sklearn.metrics import accuracy_score
from softmax_network import NeuralNetwork
import display

# ----------------- load mnist data
mnist = mnistdatas.MnistDataset("../datas")
mnist.load_train()
mnist.random_plot_different_digits()
mnist.random_plot_same_digits(9)

def check_gradients():
    n_samples = 30
    n_features = 50
    X = np.random.uniform(0,1,(n_samples,n_features))
    y = np.random.choice(10,n_samples)

    n_hidden = 20
    network = Network(n_features,n_hidden,n_output=10,l2=3e-3)
    
    weights = network.weights_vector()
    network.check_gradients(X,y,weights)   
    
def train_predict():
    ########################################## train
    network = NeuralNetwork(n_features = mnist.Xtrain.shape[1],n_hidden=50,n_output=10,l2=1e-4)
    network.fit(mnist.Xtrain,mnist.ytrain,maxiter=100)

    ########################################## accuracy on train data
    predicted_ytrain = network.predict(mnist.Xtrain)
    train_accuracy = accuracy_score(mnist.ytrain,predicted_ytrain)
    print "Train Accuracy: %3.2f%%"%(train_accuracy * 100)

    ########################################## accuracy on test data
    mnist.load_test()
    predicted_ytest = network.predict(mnist.Xtest)
    test_accuracy = accuracy_score(mnist.ytest,predicted_ytest)
    print "Test Accuracy: %3.2f%%"%(test_accuracy * 100) 

    ########################################## save the model
    #with open("softmax_network.pkl", 'wb') as outfile:
    #    cPickle.dump(network,outfile)

    ########################################## visualize general prediction
    display.visualize_predicted_digits(mnist.Xtest,mnist.ytest,predicted_ytest)

    ########################################## visualize wrong prediction
    wrong_bindices = predicted_ytest != mnist.ytest
    wrongX = mnist.Xtest[wrong_bindices,:]
    wrong_ytrue = mnist.ytest[wrong_bindices]
    wrong_ypredict = predicted_ytest[wrong_bindices]
    display.visualize_predicted_digits(wrongX,wrong_ytrue,wrong_ypredict)