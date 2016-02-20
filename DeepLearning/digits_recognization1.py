
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use("ggplot")

import commfuncs
import MnistDataset

mnist = MnistDataset.MnistDataset("../datas")
mnist.load_train()