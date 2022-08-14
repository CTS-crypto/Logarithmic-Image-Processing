from math import log
from skimage import data
import numpy as np

def find_min_max(image,n_1,n_2,m_1,m_2):

    min_max=[]

    if (m_2-m_1)%2==0:
        for i in range(n_1,n_2):
            for j in range(m_1,m_2,2):
                if image[i][j] <= image[i][j+1]:
                    min_value = image[i][j]
                    max_value = image[i][j+1]
                else:
                    min_value = image[i][j+1]
                    max_value = image[i][j]
                min_max.append((min_value,max_value))

    else:

        for i in range(n_1,n_2):
            for j in range(m_1,m_2-1,2):
                if image[i][j] <= image[i][j+1]:
                    min_value = image[i][j]
                    max_value = image[i][j+1]
                else:
                    min_value = image[i][j+1]
                    max_value = image[i][j]
                min_max.append((min_value,max_value))

        for i in range(n_1,n_2-1,2):
            if image[i][m_2-1] <= image[i+1][m_2-1]:
                min_value = image[i][m_2-1]
                max_value = image[i+1][m_2-1]
            else:
                min_value = image[i+1][m_2-1]
                max_value = image[i][m_2-1]
            min_max.append((min_value,max_value))
        
        if (n_2-n_1)%2!=0:
            min_max.append((image[n_2-1][m_2-1],image[n_2-1][m_2-1]))

    min_value=min_max[0][0]
    max_value=min_max[0][1]
    for t in range(1,len(min_max)):
        min_value=min(min_value,min_max[t][0])
        max_value=max(max_value,min_max[t][1])

    return min_value,max_value

def emee(image,alpha,d_1,d_2):
    inner_sum = 0
    k_1=len(image)//d_1
    k_2=len(image[0])//d_2
    for i in range(0,len(image),d_1):
        for j in range(0,len(image[0]),d_2):
            min_value,max_value=find_min_max(image,i,i+d_1,j,j+d_2)
            inner_sum+=((max_value+0.5)/(min_value+0.5))**alpha*log((max_value+0.05)/(min_value+0.5))
    return alpha/(k_1*k_2)*inner_sum

camera=np.array(data.camera().tolist())
moon=np.array(data.moon().tolist())

plus=camera+moon

for i in range(len(plus)):
    for j in range(len(plus[0])):
        if plus[i][j]>255:
            plus[i][j]=255


from lip import LIPSpace

print(emee(camera,1,4,4))
print(emee(moon,1,4,4))
print(emee(LIPSpace().sum(camera,moon),1,4,4))
print(emee(plus,1,4,4))
print(emee(data.camera() + data.moon(),1,4,4))