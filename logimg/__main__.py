import matplotlib.pyplot as plt
from skimage import io
from .models.lip import LIPImage,LIPSpace
from skimage.filters import scharr

a=io.imread('000000038118.jpg')
b=io.imread('CXR7_IM-2263-1001.png')

plt.imshow(a,cmap='gray',interpolation='nearest')
plt.show()
plt.imshow(b, cmap='gray', interpolation='nearest') 
plt.show()

js=LIPSpace()

ja=js.function(js.gray_tone(a))

sja=js.M-js.inverse_gray_tone(js.inverse_function(scharr(ja)))

plt.imshow(sja,cmap='gray',interpolation='nearest')
plt.show()

io.imsave('x.jpg',sja)

jb=js.function(js.gray_tone(b))

sjb=js.M-js.inverse_gray_tone(js.inverse_function(scharr(jb)))

plt.imshow(sjb,cmap='gray',interpolation='nearest')
plt.show()

io.imsave('b.jpg',sjb)