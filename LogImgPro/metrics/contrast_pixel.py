from typing import Tuple
from math import dist
from numpy import ndarray,array

def abs_contrast_2_pixels(f:Tuple,g:Tuple,M:int) -> float:
    d=dist((f[0],f[1]),(g[0],g[1]))
    return abs(f[2]-g[2])/d*256/M

def contrast_pixel(f:Tuple,img:ndarray,M:int,v:int) -> float:
    acc=0
    count=0
    for i in range(f[0]-v,f[0]+v+1):
        for j in range(f[1]-v,f[1]+v+1):
            if i==f[0] and j==f[1]:
                continue
            if i>=0 and i<img.shape[0] and j>=0 and j<img.shape[1]:
                acc+=abs_contrast_2_pixels(f,(i,j,img[i][j]),M)
                count+=1
    return acc/count

def contrast_img(img:ndarray,M:int,v:int) -> float:
    aux_img=array(img.tolist())
    acc=0
    count=0
    s=1
    if aux_img.shape[0]*aux_img.shape[1]>90000:
        s=3
    for i in range(0,aux_img.shape[0],s):
        for j in range(0,aux_img.shape[1],s):
            acc+=contrast_pixel((i,j,aux_img[i][j]),img,M,v)
            count+=1
    return acc/count