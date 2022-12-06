import numpy as np

def edge_detection(image:np.ndarray,filter):
    return filter(image)

def space_edge_detection(image:np.ndarray,filter,space):
    aux_image=space.function(space.gray_tone(image))
    return space.inverse_gray_tone(space.inverse_function(filter(aux_image)))

def parameterized_space_edge_detection(image:np.ndarray,filter,space,*,lambd=None,mu=None):
    aux_image=space.gray_tone(image)
    edge_image=space.inverse_function(filter(space.function(aux_image,lambd)),lambd)
    if mu is not None:
        return space.inverse_gray_tone(edge_image,mu)
    else:
        return space.inverse_gray_tone(edge_image)
