#script used to create confusion matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

binary1 = np.array([[28, 1],
                   [11, 32]])

fig, ax = plot_confusion_matrix(conf_mat=binary1)
plt.show()