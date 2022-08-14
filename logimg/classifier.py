import os
from skimage import io

#aqui pones la dimension 1
d1=248
#aqui pones la dimension 2
d2=184

path=os.getcwd()

name=f'{d1} x {d2}'

if not os.path.isdir(name):
    os.mkdir(name)

with os.scandir(path) as objects:
    for obj in objects:
        if obj.is_file():
            #agrega otros formatos si hay mas
            if obj.name.endswith(("jpg","JPG","png","PNG")):
                image = io.imread(obj.path)
                if image.shape==(d1,d2,3):
                    io.imsave(f'./{name}/{obj.name}',image)