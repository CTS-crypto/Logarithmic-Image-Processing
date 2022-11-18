from skimage import data,io
import matplotlib.pyplot as plt
import numpy as np
from .metrics.contrast_pixel import contrast_img
from .models.slip import SLIPSpace
from .algorithms.affine_transform import space_affine_transform

a=data.moon()

ss=SLIPSpace()

eps=0.00001

ta=np.maximum(eps,a)

ta=2*(ta-128)

simg_eq_a = space_affine_transform(a,-256,256,ss)

simg_eq_a=simg_eq_a/2+128

io.imsave('simg_eq_a.png',simg_eq_a)

simg_eq_a=io.imread('simg_eq_a.png')


print(contrast_img(simg_eq_a,256,2))
print(contrast_img(a,256,2))