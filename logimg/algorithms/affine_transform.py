import numpy as np
from math import sqrt

def affine_transform(image:np.ndarray,a:int,b:int):
    aux_image=np.array(image.tolist())
    sigma=sqrt((b-a)**2/12)
    mean_image=np.mean(aux_image)
    variance_image=sqrt(np.var(aux_image))
    return sigma/variance_image*(aux_image-mean_image)

def space_affine_transform(image:np.ndarray,a:int,b:int,space):
    aux_image=space.gray_tone(image)
    sigma=sqrt((b-a)**2/12)
    mean_image=np.mean(aux_image)
    variance_image=sqrt(np.var(aux_image))
    return space.inverse_gray_tone(space.s_mul(space.sub(aux_image,mean_image),sigma/variance_image))