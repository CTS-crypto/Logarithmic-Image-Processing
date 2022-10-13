import numpy as np
from models.logimg import LogSpace

def bhe(image:np.ndarray,X_min=0,X_max=255)->np.ndarray:
    median=int(np.median(image))
    leq_median=0
    g_median=0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j]<=median:
                leq_median+=1
            else:
                g_median+=1
    
    leq_cumulative_distribution=np.histogram(image,bins=median+1,range=(X_min,median))[0]/leq_median
    for i in range(1,len(leq_cumulative_distribution)):
        leq_cumulative_distribution[i]+=leq_cumulative_distribution[i-1]

    g_cumulative_distribution=np.histogram(image,bins=X_max-median,range=(median+1,X_max))[0]/g_median
    for i in range(1,len(g_cumulative_distribution)):
        g_cumulative_distribution[i]+=g_cumulative_distribution[i-1]
    
    new_image=np.zeros(image.shape)
    
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            if image[i,j]<=median:
                new_image[i,j]=X_min+leq_cumulative_distribution[image[i,j]]*(median-X_min)
            else:
                new_image[i,j]=median+g_cumulative_distribution[image[i,j]-median-1]*(X_max-median)
    
    return new_image

def space_bhe(image:np.ndarray,space:LogSpace,X_min=0,X_max=255)->np.ndarray:
    neg_image=space.M-image
    median=int(np.median(neg_image))
    leq_median=0
    g_median=0
    for i in range(neg_image.shape[0]):
        for j in range(neg_image.shape[1]):
            if neg_image[i,j]<=median:
                leq_median+=1
            else:
                g_median+=1
    
    leq_cumulative_distribution=np.histogram(neg_image,bins=median+1,range=(X_min+1,median))[0]/leq_median
    for i in range(1,len(leq_cumulative_distribution)):
        leq_cumulative_distribution[i]+=leq_cumulative_distribution[i-1]

    g_cumulative_distribution=np.histogram(neg_image,bins=X_max+1-median,range=(median+1,X_max+1))[0]/g_median
    for i in range(1,len(g_cumulative_distribution)):
        g_cumulative_distribution[i]+=g_cumulative_distribution[i-1]
    
    new_image=np.zeros(neg_image.shape)
    
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            if neg_image[i,j]<=median:
                new_image[i,j]=space.sum(X_min,space.s_mul(space.sub(median,X_min),leq_cumulative_distribution[neg_image[i,j]]))
            else:
                new_image[i,j]=space.sum(median,space.s_mul(space.sub(X_max,median),g_cumulative_distribution[neg_image[i,j]-median-1]))
    
    return space.neg(new_image)