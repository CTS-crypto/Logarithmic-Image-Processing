import numpy as np
from math import sqrt

def space_affine_transform(image:np.ndarray,a:int,b:int,space):
    aux_image=space.gray_tone(image)
    sigma=sqrt((b-a)**2/12)
    mean_image=np.mean(aux_image)
    variance_image=sqrt(np.var(aux_image))
    return space.inverse_gray_tone(space.s_mul(space.sub(aux_image,mean_image),sigma/variance_image))