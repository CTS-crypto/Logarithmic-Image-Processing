from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from .algorithms.affine_transform import space_affine_transform
from .models.lip import LIPSpace
from .models.hlip import HLIPSpace
from .models.pslip import PSLIPSpace
from .models.slip import SLIPSpace


m=data.moon()

print(np.min(m),np.max(m))

plt.imshow(m,cmap='gray',interpolation='nearest')

plt.show()

plt.hist(m.ravel(),bins=256)

plt.show()

m1=space_affine_transform(m,-256,256,SLIPSpace())

print(np.min(m1),np.max(m1))

plt.imshow(m1,cmap='gray',interpolation='nearest')

plt.show()

plt.hist(m1.ravel(),bins=256)

plt.show()