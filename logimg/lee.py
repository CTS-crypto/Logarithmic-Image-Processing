import numpy as np
from lip import LIPSpace
from logimg import LogSpace

def lee(image:np.ndarray,n:int,theta,rho,xi)->np.ndarray:
    aux_image=np.array(image.tolist())
    a=np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            a[i,j]=np.mean(aux_image[max(i-n,0):min(i+n+1,aux_image.shape[1]),max(j-n,0):min(j+n+1,aux_image.shape[1])])
    return theta*a+rho+xi*(aux_image-a)

def space_lee(image:np.ndarray,n:int,eta,sigma,delta,space:LogSpace)->np.ndarray:
    aux_image=space.M-1-image
    a=np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            a[i,j]=np.mean(aux_image[max(i-n,0):min(i+n+1,aux_image.shape[1]),max(j-n,0):min(j+n+1,aux_image.shape[1])])
    
    return space.neg(space.sum(space.sum(space.s_mul(a,eta),sigma),space.s_mul(space.sub(aux_image,a),delta)))      

from skimage import data
import matplotlib.pyplot as plt

m=data.camera()

plt.imshow(m,cmap='gray',interpolation='nearest')
plt.show()

l=lee(m,4,0.4,200,0.19)
plt.imshow(l,cmap='gray',interpolation='nearest')
plt.show()

space_l=space_lee(m,4,0.4,200,0.19,LIPSpace())
plt.imshow(space_l,cmap='gray',interpolation='nearest')
plt.show()