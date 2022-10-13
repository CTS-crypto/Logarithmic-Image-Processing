import os
from skimage import io
from PIL import ImageFile
import json
from models.plip import PLIPSpace
from emee import emee
import numpy as np

count=0

def sumit(path:str):
    no_path={}

    name=path + "/sum"
    if not os.path.isdir(name):
        os.mkdir(name)

    image_list=[]
    with os.scandir(path) as objects:
        for obj in objects:
            if obj.name.endswith(("jpg","JPG","png","PNG")):
                image=io.imread(obj.path)
                no_path[len(image_list)]=obj.path
                image_list.append((space.M-image,obj.path))


    with open(f'{path}/no_path.json','w') as fout:
        json.dump(no_path, fout)

    for i in range(len(image_list)):
        
        with open('test_img_sum.json','r') as fout:
            img_sum=json.load(fout)
        
        for j in range(i+1,len(image_list)):
            d_1 = 4 if image_list[i][0].shape[0]%4==0 else 5
            d_2 = 4 if image_list[i][0].shape[1]%4==0 else 5
            max_emee=-1
            max_gamma=-1
            max_mi=-1
            for ga in gammas:
                result_sum=space.sum(image_list[i][0],image_list[j][0],ga)
                for mi in mis:
                    max_pi_r=np.max(result_sum)
                    print(ga,mi)
                    if mi <= max_pi_r:
                        continue
                    result=space.neg(result_sum,mi)
                    actual_emee=emee(result,1,d_1,d_2)
                    if actual_emee > max_emee:
                        max_emee=actual_emee
                        max_gamma=ga
                        max_mi=mi
            result=space.neg(space.sum(image_list[i][0],image_list[j][0],max_gamma),max_mi)
            io.imsave(f'{name}/sum_{i}_{j}.jpg',result)
            img_sum[f'{image_list[i][1]}&{image_list[j][1]}']=(max_gamma,max_mi)
            print(image_list[i][1],image_list[j][1],max_gamma,max_mi)
        
        with open('test_img_sum.json','w') as fout: 
            json.dump(img_sum,fout) 

def dfs(entry:os.DirEntry):
    with os.scandir(entry.path) as objects:
        for obj in objects:
            if obj.is_dir():
                if obj.name=='Train':
                    continue
                else:
                    dfs(obj)
            elif obj.name.endswith(("jpg","JPG","png","PNG")):
                sumit(obj.path.replace(obj.name,""))
                break

space=PLIPSpace()

mis=[256,400,500,800,1000,1026,4100,600,300,250,200]
gammas=[256,400,1000,2000,300,500,600,512,1026,4100,1280,800,526,200]
betas=[1,5,10,19,1.1,1.2,1.5,0.45]
lambdas=[256,-500,-1026,-1000,800,512,1026,4100,600,200]
ks=[256,600,256,512,1026,500,300,800,4100,200]

ImageFile.LOAD_TRUNCATED_IMAGES=True

path=os.getcwd()

with os.scandir(path) as objects:
    for obj in objects:
        if obj.is_dir() and obj.name.startswith("gray"):
                dfs(obj)