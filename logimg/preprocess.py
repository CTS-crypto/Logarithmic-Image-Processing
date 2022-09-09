import os
from skimage import io,color
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True

path=os.getcwd()

with os.scandir(path) as objects:
    for obj in objects:
        if obj.is_dir():
            if obj.name.startswith(("_",".","gray")):
                continue
            name=f'gray {obj.name}'
            if not os.path.isdir(name):
                os.mkdir(name)
            print(name)
            with os.scandir(obj.path) as images:
                for img in images:
                    image = io.imread(img.path)
                    if len(image.shape)==2:
                        gray_image=image
                    else:
                        gray_image = color.rgb2gray(image)
                    io.imsave(f'./{name}/{img.name}',gray_image)