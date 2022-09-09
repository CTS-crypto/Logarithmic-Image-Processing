import os
from skimage import io
from PIL import ImageFile
import json
from plip import PLIPSpace
from emee import emee
import numpy as np

def dfs(entry:os.DirEntry):
    with os.scandir(entry.path) as objects:
        for obj in objects:
            if obj.is_dir():
                dfs(obj)
            elif obj.name.endswith(("jpg","JPG","png","PNG")):
                image = io.imread(obj.path)
                d_1 = 4 if image.shape[0]%4==0 else 5
                d_2 = 4 if image.shape[1]%4==0 else 5
                max_emee=-1
                max_mi=-1
                for mi in mis:
                    if mi <= np.max(image):
                        continue
                    result=space.neg(image,mi)
                    actual_emee=emee(result,1,d_1,d_2)
                    if actual_emee > max_emee:
                        max_emee=actual_emee
                        max_mi=mi
                img_mi[obj.path]=max_mi
                print(obj.path,max_mi)

space=PLIPSpace()

mis=[256,400,500,800,1000,1026,4100,600,300,250,200]
gammas=[256,400,1000,2000,300,500,600,512,1026,4100,1280,800,526,200]
betas=[1,5,10,19,1.1,1.2,1.5,0.45]
lambdas=[256,-500,-1026,-1000,800,512,1026,4100,600,200]
ks=[256,600,256,512,1026,500,300,800,4100,200]

ImageFile.LOAD_TRUNCATED_IMAGES=True


img_mi={}

path=os.getcwd()

with os.scandir(path) as objects:
    for obj in objects:
        if obj.is_dir() and obj.name.startswith("gray"):
                dfs(obj)

with open('img_mi.json','w') as fout:
    json.dump(img_mi, fout)
