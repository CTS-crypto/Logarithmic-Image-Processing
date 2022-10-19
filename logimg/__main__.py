from .models.hlip import HLIPImage
from .algorithms.basic_operations import space_mul_img
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from .emee import emee
from skimage import io

a=io.imread('CXR7_IM-2263-1001.png',as_gray=True)

print(a)

plt.imshow(a,cmap='gray',interpolation='nearest')
plt.show()