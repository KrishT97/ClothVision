import numpy as np
from matplotlib import pyplot as plt
import scipy.io
mat = scipy.io.loadmat('./data/labels_raw/pixel_level_labels_mat/0001.mat')

print(mat.items())
plt.imshow(mat["groundtruth"])
plt.show()