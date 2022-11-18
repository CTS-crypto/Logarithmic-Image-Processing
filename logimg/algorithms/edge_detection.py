import numpy as np

def edge_detection(image:np.ndarray,filter):
    return filter(image)

def space_edge_detection(image:np.ndarray,filter,space):
    aux_image=space.function(space.gray_tone(image))
    return space.inverse_gray_tone(space.inverse_function(filter(aux_image)))