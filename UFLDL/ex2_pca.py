
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from pca import PCA

def demo_pca_2d():
    datafile = "matlab/pcaData.txt"
    X = np.loadtxt(datafile).T

    pca = PCA(whiten=True)
    Xwhiten = pca.fit_transform(X)

    plt.scatter(X[:,0],X[:,1])# in original X, each column is a feature
    # each column in V is a new axis
    for i in xrange(2):
        newaxis = pca.V[:,i]
        pntX = [-newaxis[0],newaxis[0]]
        pntY = [-newaxis[1],newaxis[1]]
        plt.plot(pntX,pntY,label="%d principle component"%(i+1))
    plt.legend(loc="best")
    plt.title("original datas and princple components")

    plt.scatter(pca.Xprojected[:,0],pca.Xprojected[:,1])
    plt.title("projected, axes have different scales")

    plt.scatter(Xwhiten[:,0],Xwhiten[:,1])
    plt.title("whitened, all axes have similar scales")








