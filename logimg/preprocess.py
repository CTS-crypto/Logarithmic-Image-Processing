import os
from skimage import io,color
import matplotlib.pyplot as plt

path=os.getcwd()

with os.scandir(path) as objects:
    for obj in objects:
        if obj.is_dir():
            if obj.name.startswith(("_",".","gray")):
                continue
            name=f'gray {obj.name}'
            if not os.path.isdir(name):
                os.mkdir(name)
            with os.scandir(obj.path) as images:
                for img in images:
                    image = io.imread(img.path)
                    gray_image = color.rgb2gray(image)
                    io.imsave(f'./{name}/{img.name}',gray_image)