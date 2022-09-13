import numpy as np
import matplotlib.pyplot as plt
from lip import LIPSpace
from logimg import LogSpace

def alpha_rooting(image,alpha):
    t_image=np.fft.fft2(image)
    t_new_image=t_image*(image+0.005)**(alpha-1)
    new_image=np.abs(np.fft.ifft2(t_new_image))
    return new_image

def space_alpha_rooting(image,alpha,space:LogSpace):
    neg_image=space.M-1-image
    t_image=np.fft.fft2(neg_image)
    t_new_image=space.inverse_equation((space.equation(neg_image)+1)**(alpha-1))*t_image
    new_image=space.neg(np.abs(np.fft.ifft2(t_new_image)))
    return new_image

from skimage import data

m=data.camera()

plt.imshow(m,cmap='gray',interpolation='nearest')
plt.show()

b=alpha_rooting(m,0.98)
plt.imshow(b,cmap='gray',interpolation='nearest')
plt.show()

spa=space_alpha_rooting(m,0.98,LIPSpace())
plt.imshow(spa,cmap='gray',interpolation='nearest')
plt.show()