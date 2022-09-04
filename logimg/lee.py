import numpy as np
from lip import LIPImage

def lee(image:np.ndarray,n:int,theta,rho,xi)->np.ndarray:
    aux_image=np.array(image.tolist())
    a=np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            a[i,j]=np.mean(aux_image[max(i-n,0):min(i+n+1,aux_image.shape[1]),max(j-n,0):min(j+n+1,aux_image.shape[1])])
    return theta*a+rho+xi*(aux_image-a)

def space_lee(image:np.ndarray,n:int,eta,sigma,delta,type_image,M=256)->np.ndarray:
    aux_image=type_image(image,M)
    aux_image2=type_image(image,M)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            aux_image2[i][j]=np.mean(aux_image[max(i-n,0):min(i+n+1,aux_image.shape[1]),max(j-n,0):min(j+n+1,aux_image.shape[1])])
    return (aux_image2*eta+sigma+(aux_image-aux_image2)*delta).transform()            

from skimage import data
import matplotlib.pyplot as plt

m=data.camera()

plt.imshow(m,cmap='gray',interpolation='nearest')
plt.show()

l=lee(m,4,0.42,450,0.2)
plt.imshow(l,cmap='gray',interpolation='nearest')
plt.show()

space_l=space_lee(m,4,0.42,450,0.2,LIPImage)
plt.imshow(space_l,cmap='gray',interpolation='nearest')
plt.show()