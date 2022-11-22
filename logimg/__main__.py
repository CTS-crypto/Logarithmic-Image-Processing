from skimage import data,io
import matplotlib.pyplot as plt
import numpy as np
from .metrics.contrast_pixel import contrast_img
from .models.slip import SLIPSpace
from .algorithms.affine_transform import space_affine_transform

a=data.moon()

ss=SLIPSpace()

print(ss.function(120)*30)

print(ss.function(ss.s_mul(120,30)))