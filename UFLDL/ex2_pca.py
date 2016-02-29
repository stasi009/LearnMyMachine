
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from pca import PCA

datafile = "matlab/pcaData.txt"

X = np.loadtxt(datafile).T
# plt.scatter(X[:,0],X[:,1],marker="o",s=30,facecolors='none')
plt.scatter(X[:,0],X[:,1])

pca = PCA(whiten=True)
Xwhiten = pca.fit_transform(X)
plt.plot([0,pca.V[0,0]],[0,pca.V[1,0]])
plt.plot([0,pca.V[0,1]],[0,pca.V[1,1]])


plt.scatter(Xwhiten[:,0],Xwhiten[:,1])
plt.scatter(pca.Xprojected[:,0],pca.Xprojected[:,1])


pca = PCA(n_components=1, whiten=True)
Xwhiten = pca.fit_transform(X)
plt.scatter(Xwhiten[:,0],Xwhiten[:,1])
plt.scatter(pca.Xprojected[:,0],pca.Xprojected[:,1])





