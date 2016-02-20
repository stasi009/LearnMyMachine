
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use("ggplot")

import commfuncs

DataPath = "../datas"
ytrain,Xtrain = commfuncs.load_mnist(DataPath,"train")
        
def random_plot_digits():
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True,sharey=True)
    axes = axes.flatten()

    for index in xrange(10):
        allimages = Xtrain[ytrain == index]
        imag = allimages[ np.random.randint(0,len(allimages))].reshape(28,28)
        axes[index].imshow(imag, cmap='Greys', interpolation='nearest')

    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.tight_layout()
    plt.show()
