import numpy as np
import matplotlib.pyplot as plt
from lip import LIPSpace
from logimg import LogSpace

def alpha_rooting(image,alpha):
    t_image=np.fft.fft2(image)
    t_new_image=t_image*np.abs(t_image+0.005)**(alpha-1)
    new_image=np.abs(np.fft.ifft2(t_new_image))
    return new_image

def space_alpha_rooting(image,alpha,type_space,M):
    space : LogSpace =type_space(M)
    t_image=np.fft.fft2(space.neg(image))
    abs_t_image=np.abs(t_image)
    abs_t_max=np.max(abs_t_image)
    aux_space : LogSpace =type_space(abs_t_max+1)
    t_new_image=aux_space.inverse_equation((aux_space.equation(abs_t_image)+1)**(alpha-1))*t_image
    new_image=np.abs(np.fft.ifft2(t_new_image))
    return space.neg(new_image)

from skimage import data

m=data.camera()

plt.imshow(m,cmap='gray',interpolation='nearest')
plt.show()

b=alpha_rooting(m,0.98)
plt.imshow(b,cmap='gray',interpolation='nearest')
plt.show()

spa=space_alpha_rooting(m,0.98,LIPSpace,800)
plt.imshow(spa,cmap='gray',interpolation='nearest')
plt.show()