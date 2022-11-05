import numpy as np
import matplotlib.pyplot as plt
from ..models.logimg import LogSpace

def alpha_rooting(image,alpha):
    t_image=np.fft.fft2(image)
    t_new_image=t_image*(np.abs(t_image))**(alpha-1)
    new_image=np.abs(np.fft.ifft2(t_new_image))
    return new_image

def space_alpha_rooting(image,alpha,space:LogSpace):
    t_image=np.fft.fft2(image)
    abs_t_image=np.abs(t_image)
    space.M=np.max(abs_t_image)
    t_new_image=space.inverse_function(space.function(abs_t_image)**(alpha-1))*t_image
    new_image=np.abs(np.fft.ifft2(t_new_image))
    return new_image