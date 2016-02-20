
import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def load_mnist(path,prefix):
    labels_path = os.path.join(path,"%s-labels.idx1-ubyte"%prefix)
    with open(labels_path,"rb") as label_file:
        label_file.read(8)
        labels = np.fromfile(label_file,dtype=np.uint8)

    images_path = os.path.join(path,"%s-images.idx3-ubyte"%prefix)
    with open(images_path,"rb") as image_file:
        image_file.read(16)
        # each image is 28*28=784 pixels
        images = np.fromfile(image_file,dtype=np.uint8).reshape(len(labels),784)

    return labels,images

def plot_digits():
    pass