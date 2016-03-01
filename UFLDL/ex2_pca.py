
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import matloader
import display
from pca import SimplePCA

def demo_image_pca():
    seed = 9
    np.random.seed(seed)

    n_samples =10000
    n_selected =400
    random_sel = np.random.choice(n_samples,n_selected)
        
    # --------------------- load and show original images
    patches = matloader.sample_raw_images("matlab/IMAGES_RAW.mat",n_samples,seed)
    images = patches.T # [n_samples,n_features], each row is a image, each column will be a pixel
    display.display_image_patches(images[random_sel,:],filename="raw_images.png",direction="byrow")

    # --------------------- zero mean
    image_means = images.mean(axis=1)# mean for each sample
    images = images - image_means[:,np.newaxis]# transform into column vector to be broadcasted
    display.display_image_patches(images[random_sel,:],filename="raw_images_0mean.png",direction="byrow")

    # --------------------- SVD
    U,S,Vt = np.linalg.svd(images)
    V = Vt.T

    # --------------------- check the covariance
    image_rotated = images.dot(V)
    covariance_rotated = image_rotated.T.dot(image_rotated)
    plt.imshow(covariance_rotated)

    # --------------------- whitening
    epsilon = 1e-6
    image_whiten = image_rotated / (S+epsilon)
    covariance_whiten = image_whiten.T.dot(image_whiten)
    plt.imshow(covariance_whiten)

    # --------------------- find the best k
    def reconstruct(var_kept_percent):
        explained_var = S**2
        explained_var_ratios = explained_var / explained_var.sum()
        var_ratios_cumsum = explained_var_ratios.cumsum()
        K = np.sum(var_ratios_cumsum < (var_kept_percent/100.0)) + 1
        print "keep %d principle components to keep more than %d%% variance"%(K,var_kept_percent)

        constructed_images = images.dot(V[:,:K]).dot(V[:,:K].T)

        imgfile = "images_reconstruct_%dvariance.png"%(var_kept_percent)
        display.display_image_patches(constructed_images[random_sel,:],filename=imgfile,direction="byrow")

    for percent in [50,90,95,99]:
        reconstruct(percent)

def demo_pca_2d():
    datafile = "matlab/pcaData.txt"
    X = np.loadtxt(datafile).T

    pca = SimplePCA(whiten=True)
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

def simple_pca_reconstruct():
    datafile = "matlab/pcaData.txt"
    X = np.loadtxt(datafile).T

    pca = SimplePCA(n_components=1,whiten=False)
    _ = pca.fit_transform(X)

    reconstructed = pca.reconstruct()
    plt.plot(reconstructed[:,0],reconstructed[:,1])









