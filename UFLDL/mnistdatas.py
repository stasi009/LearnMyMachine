
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

class MnistDataset(object):
    
    def __init__(self,path,toscale=True):
        self.path = path
        self.toscale = toscale

    def load_train(self):
        self.Xtrain,self.ytrain = self.__load("train")

    def load_test(self):
        self.Xtest,self.ytest = self.__load("t10k")

    def __load(self,prefix):
        labels_path = os.path.join(self.path,"%s-labels.idx1-ubyte" % prefix)
        with open(labels_path,"rb") as label_file:
            label_file.read(8)
            labels = np.fromfile(label_file,dtype=np.uint8)

        images_path = os.path.join(self.path,"%s-images.idx3-ubyte" % prefix)
        with open(images_path,"rb") as image_file:
            image_file.read(16)
            # each image is 28*28=784 pixels
            images = np.fromfile(image_file,dtype=np.uint8).reshape(len(labels),784)

        if self.toscale: return images / 255.0,labels
        else: return images,labels

    def random_plot_different_digits(self):
        fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True,sharey=True)
        axes = axes.flatten()

        for index in xrange(10):
            allimages = self.Xtrain[self.ytrain == index]
            imag = allimages[np.random.randint(0,allimages.shape[0])].reshape(28,28)
            axes[index].imshow(imag, cmap='Greys', interpolation='nearest')

        axes[0].set_xticks([])
        axes[0].set_yticks([])
        plt.tight_layout()
        plt.show()

    def random_plot_same_digits(self,digit,nrows=4,ncols=4):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,sharey=True)
        axes = axes.flatten()

        all_valid_images = self.Xtrain[self.ytrain == digit]
        selected_images = all_valid_images[np.random.randint(0,all_valid_images.shape[0],nrows * ncols)]

        for index in xrange(selected_images.shape[0]):
            imag = selected_images[index].reshape(28,28)
            axes[index].imshow(imag, cmap='Greys', interpolation='nearest')

        axes[0].set_xticks([])
        axes[0].set_yticks([])
        plt.tight_layout()
        plt.show()




