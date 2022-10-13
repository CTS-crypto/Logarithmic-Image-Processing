import numpy as np
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import json

dimention=700

with open('train_img_sum.json','r') as fout:
    train_img_sum=json.load(fout)

with open('test_img_sum.json','r') as fout:
    test_img_sum=json.load(fout)

x_train=[]
y_train=[]
x_test=[]
y_test=[]

for item in train_img_sum:
    imgs=item.split('&')
    img1,img2=io.imread(imgs[0]),io.imread(imgs[1])
    if img1.shape[0] > 1000:
        img1=resize(img1,(img1.shape[0]//2,img2.shape[1]//2),anti_aliasing=True)
        img2=resize(img2,(img2.shape[0]//2,img2.shape[1]//2),anti_aliasing=True)
    pad_img1=np.pad(img1,[[(dimention-img1.shape[0])//2,(dimention-img1.shape[0])//2],[(dimention-img1.shape[1])//2,(dimention-img1.shape[1])//2]],mode='constant',constant_values=0)
    pad_img2=np.pad(img2,[[(dimention-img2.shape[0])//2,(dimention-img2.shape[0])//2],[(dimention-img2.shape[1])//2,(dimention-img2.shape[1])//2]],mode='constant',constant_values=0)
    con=np.concatenate((pad_img1,pad_img2))
    x_train.append(con)
    y_train.append(train_img_sum[item])

print(1)

for item in test_img_sum:
    imgs=item.split('&')
    img1,img2=io.imread(imgs[0]),io.imread(imgs[1])
    if img1.shape[0] > 1000:
        img1=resize(img1,(img1.shape[0]//2,img2.shape[1]//2),anti_aliasing=True)
        img2=resize(img2,(img2.shape[0]//2,img2.shape[1]//2),anti_aliasing=True)
    pad_img1=np.pad(img1,[[(dimention-img1.shape[0])//2,(dimention-img1.shape[0])//2],[(dimention-img1.shape[1])//2,(dimention-img1.shape[1])//2]],mode='constant',constant_values=0)
    pad_img2=np.pad(img2,[[(dimention-img2.shape[0])//2,(dimention-img2.shape[0])//2],[(dimention-img2.shape[1])//2,(dimention-img2.shape[1])//2]],mode='constant',constant_values=0)
    con=np.concatenate((pad_img1,pad_img2))
    x_test.append(con)
    y_test.append(test_img_sum[item])

print(2)

plt.imshow(x_train[0], cmap='gray', interpolation='nearest')
plt.show()