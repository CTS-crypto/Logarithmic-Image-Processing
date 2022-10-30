from .emee import emee
from .models.plip import PLIPSpace
from skimage import io
from skimage.filters import scharr
import numpy as np
import matplotlib.pyplot as plt

mis=[256,400,500,800,1000,1026,4100,600,300,250,200]
gammas=[256,400,1000,2000,300,500,600,512,1026,4100,1280,800,526,200]

a=io.imread('000000038118.jpg',as_gray=True)
b=io.imread('CXR7_IM-2263-1001.png',as_gray=True)

ps=PLIPSpace()

d_1 = 4 if a.shape[0]%4==0 else 5
d_2 = 4 if a.shape[1]%4==0 else 5
max_emee=-1
max_gamma=-1
max_mi=-1
for ga in gammas:
    for mi in mis:
        pa=ps.equation(ps.neg(a,mi),ga)
        pa_spa=ps.neg(ps.inverse_equation(pa+scharr(pa),ga))
        max_pi_r=np.max(pa_spa)
        if mi < max_pi_r:
            continue
        result=ps.neg(pa_spa,mi)
        actual_emee=emee(result,1,d_1,d_2,0.5)
        if actual_emee > max_emee:
            max_emee=actual_emee
            max_gamma=ga
            max_mi=mi
pa=ps.equation(ps.neg(a,max_mi),max_gamma)
result=ps.neg(ps.inverse_equation(pa+scharr(pa),max_gamma),max_mi)
print(max_gamma,max_mi)

print(result)
plt.imshow(result,cmap='gray',interpolation='nearest')
plt.show()