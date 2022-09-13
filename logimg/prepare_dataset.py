from email.mime import image
import os
from skimage import io
from PIL import ImageFile
import json
from plip import PLIPSpace
from emee import emee
import numpy as np

count=0

def sumit(entry:os.DirEntry):
    no_path={}

    name=entry.path.replace("neg","")+"sum"
    if not os.path.isdir(name):
        os.mkdir(name)

    image_list=[]
    with os.scandir(entry.path) as objects:
        for obj in objects:
            if obj.name.endswith(("jpg","JPG","png","PNG")):
                image=io.imread(obj.path)
                no_path[len(image_list)]=obj.path
                image_list.append((image,obj.path))

    with open(f'{entry.path}/no_path.json','w') as fout:
        json.dump(no_path, fout)

    for i in range(len(image_list)):
        for j in range(i,len(image_list)):
            d_1 = 4 if image_list[i][0].shape[0]%4==0 else 5
            d_2 = 4 if image_list[i][0].shape[1]%4==0 else 5
            max_emee=-1
            max_gamma=-1
            for ga in gammas:
                result=space.sum(image_list[i][0],image_list[j][0],ga)
                actual_emee=emee(result,1,d_1,d_2)
                if actual_emee > max_emee:
                    max_emee=actual_emee
                    max_gamma=ga
            result=space.sum(image_list[i][0],image_list[j][0],max_gamma)
            print(np.max(result))
            io.imsave(f'{name}/sum_{i}_{j}.jpg',result)
            img_mi[f'{image_list[i][1]}&{image_list[j][1]}']=max_gamma
            #print(image_list[i][1],image_list[j][1],max_gamma)

def dfs(entry:os.DirEntry):
    with os.scandir(entry.path) as objects:
        for obj in objects:
            if obj.is_dir():
                if obj.name=='Test':
                    continue
                if obj.name=='sum':
                    with os.scandir(obj.path) as images:
                        for i in images:
                            image=io.imread(i.path)
                            print(np.max(image))
                else:
                    dfs(obj)

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

with open('train_img_sum.json','w') as fout:
    json.dump(img_mi, fout)
