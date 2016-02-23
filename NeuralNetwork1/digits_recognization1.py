
import cPickle
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use("ggplot")
from sklearn.metrics import accuracy_score

import commfuncs
import mnistdatas
import neuralnetwork1

########################################## load train data
mnist = mnistdatas.MnistDataset("../datas")
mnist.load_train()
mnist.random_plot_different_digits()
mnist.random_plot_same_digits(9)

########################################## enable gradient checking
#params = {}
#params["l2"] = 0.1
#params["epochs"] = 10 # don't need so many epoches when perform gradient checking
#params["minibatches"] = 3
#params["learnrate"] = 0.001
#params["shrink_learnrate"] = 0.00001
#params["shrink_velocity"] = 0.001
#params["shuffle"] = False
#params["checkgrad_epsilon"] = 1e-5 # enable gradient checking

## only for debugging/test purpose, no need to specify many hidden neurons
#network = neuralnetwork1.NeuralNetwork(n_features = mnist.Xtrain.shape[1],n_hidden=2,n_digits=10)
#network.fit(mnist.Xtrain,mnist.ytrain,params)

########################################## train
network = neuralnetwork1.NeuralNetwork(n_features = mnist.Xtrain.shape[1],n_hidden=50,n_digits=10)

params = {}
params["l2"] = 0.1
params["epochs"] = 1000
params["minibatches"] = 50
params["learnrate"] = 0.001
params["shrink_learnrate"] = 0.00001
params["shrink_velocity"] = 0.001
params["shuffle"] = True

costs = network.fit(mnist.Xtrain,mnist.ytrain,params)
plt.plot(costs)

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
with open("network.pkl", 'wb') as outfile:
    cPickle.dump(network,outfile)

########################################## visualize the prediction
def visualize_prediction(X,ytrue,ypredict,nrows=5,ncols=5):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,sharey=True)
    axes = axes.flatten()

    selected_indices = np.random.randint(0,X.shape[0],nrows*ncols)
    selected_images = X[selected_indices,:]
    selected_ytrue = ytrue[selected_indices]
    selected_ypredict = ypredict[selected_indices]

    for index in xrange(nrows*ncols):
        imag = selected_images[index].reshape(28,28)
        axes[index].imshow(imag, cmap='Greys', interpolation='nearest')
        title = axes[index].set_title('(%d) t: %d p: %d'% (index+1, selected_ytrue[index], selected_ypredict[index]))

        # set title color based prediction is right or wrong
        color = "b" if selected_ytrue[index] == selected_ypredict[index] else "r"
        plt.setp(title, color=color)   

    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.tight_layout()
    plt.show()

visualize_prediction(mnist.Xtest,mnist.ytest,predicted_ytest)

########################################## visualize wrong prediction
wrong_bindices = predicted_ytest != mnist.ytest
wrongX = mnist.Xtest[wrong_bindices,:]
wrong_ytrue = mnist.ytest[wrong_bindices]
wrong_ypredict = predicted_ytest[wrong_bindices]
visualize_prediction(wrongX,wrong_ytrue,wrong_ypredict)



