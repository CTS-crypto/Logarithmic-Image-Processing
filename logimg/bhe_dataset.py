import os
from skimage import io
from PIL import ImageFile
import json
from models.plip import PLIPSpace
from emee import emee
import numpy as np
from algorithms.bhe import space_bhe

count=0

def leeit(path:str):
    name=path + "/bhe"
    if not os.path.isdir(name):
        os.mkdir(name)

    print(name)

    with os.scandir(path) as objects:
        for obj in objects:
            if obj.name.endswith(("jpg","JPG","png","PNG")):
                image=io.imread(obj.path)
                
                with open('train_img_bhe.json','r') as fout:
                    img_lee=json.load(fout)
            
                d_1 = 4 if image.shape[0]%4==0 else 5
                d_2 = 4 if image.shape[1]%4==0 else 5
            
                max_emee=-1
                max_gamma=-1
                max_mi=-1
                for mi in mis:
                    for gamma in gammas:
                        space=PLIPSpace(256,mi,gamma,gamma) 
                        result=space_bhe(image,space)
                        min_pi=np.min(result)
                        if min_pi < 0:
                            continue
                        actual_emee=emee(result,1,d_1,d_2)
                        if actual_emee > max_emee:
                            max_emee=actual_emee
                            max_gamma=gamma
                            max_mi=mi
                        print(mi,gamma)
                result=space_bhe(image,PLIPSpace(256,max_mi,max_gamma,max_gamma))
                io.imsave(f'{name}/{obj.name}',result)
                img_lee[f'{obj.path}']=(max_mi,max_gamma)
                print(obj.path,max_mi,max_gamma)
            
                with open('train_img_bhe.json','w') as fout: 
                    json.dump(img_lee,fout) 

def dfs(entry:os.DirEntry):
    with os.scandir(entry.path) as objects:
        for obj in objects:
            if obj.is_dir():
                if obj.name=='Test' or obj.name=='sum':
                    continue
                else:
                    dfs(obj)
            elif obj.name.endswith(("jpg","JPG","png","PNG")):
                leeit(obj.path.replace(obj.name,""))
                break

mis=[256,400,500,800,1000,1026,4100,600,300,250,200]
gammas=[256,400,1000,2000,300,500,600,512,1026,4100,1280,800,526,200]
betas=[1,5,10,19,1.1,1.2,1.5,0.45]
lambdas=[256,-500,-1026,-1000,800,512,1026,4100,600,200]
ks=[256,600,256,512,1026,500,300,800,4100,200]

ImageFile.LOAD_TRUNCATED_IMAGES=True

path=os.getcwd()

with open('train_img_bhe.json','w') as fout: 
    json.dump({},fout) 

with os.scandir(path) as objects:
    for obj in objects:
        if obj.is_dir() and obj.name.startswith("gray"):
                dfs(obj)