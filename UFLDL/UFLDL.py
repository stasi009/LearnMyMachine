
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use("ggplot")
from sklearn.metrics import accuracy_score

import mnistdatas

########################################## load train data
mnist = mnistdatas.MnistDataset("../datas")
mnist.load_train()
mnist.random_plot_different_digits()
mnist.random_plot_same_digits(9)