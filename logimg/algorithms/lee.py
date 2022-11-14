import numpy as np
from ..models.logimg import LogSpace

def lee(image:np.ndarray,n:int,theta,rho,zeta)->np.ndarray:
    aux_image=np.array(image.tolist())
    a=np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            a[i,j]=np.mean(aux_image[max(i-n,0):min(i+n+1,aux_image.shape[1]),max(j-n,0):min(j+n+1,aux_image.shape[1])])
    return theta*a+rho+zeta*(aux_image-a)

def space_lee(image:np.ndarray,n:int,eta,sigma,delta,space:LogSpace)->np.ndarray:
    aux_image=space.function(space.gray_tone(image))
    a=np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            a[i,j]=np.mean(aux_image[max(i-n,0):min(i+n+1,aux_image.shape[1]),max(j-n,0):min(j+n+1,aux_image.shape[1])])
    return space.inverse_gray_tone(space.inverse_function(eta*a+sigma+delta*(aux_image-a)))