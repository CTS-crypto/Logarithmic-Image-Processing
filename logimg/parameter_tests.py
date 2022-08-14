from sympy import *
from skimage import data
import numpy as np

g = Symbol('g')

camera=np.array(data.camera().tolist())
moon=np.array(data.moon().tolist())

pixels=len(camera)*len(camera[0])

matrix=[[(camera[i][j]+moon[i][j]-camera[i][j]*moon[i][j]/g - (camera[i][j]+moon[i][j]-camera[i][j]*moon[i][j]/256))**2 for j in range(len(camera[0]))] for i in range(len(camera))]

L=sum([sum(i) for i in matrix])/pixels

print(L.diff(g))