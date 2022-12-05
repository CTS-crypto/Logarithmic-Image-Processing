import numpy as np

def unsharp_masking(image:np.ndarray,filter,op:str):
    if op=='+':
        return image+filter(image)
    elif op=='-':
        return image-filter(image)
    else:
        raise ValueError('Invalid operator')

def space_unsharp_masking(image:np.ndarray,filter,op:str,space):
    aux_image=space.function(space.gray_tone(image))
    if op=='+':
        return space.inverse_gray_tone(space.inverse_function(aux_image+filter(aux_image)))
    elif op=='-':
        return space.inverse_gray_tone(space.inverse_function(aux_image-filter(aux_image)))
    else:
        raise ValueError('Invalid operator')

def parameterized_space_unsharp_masking(image:np.ndarray,filter,op:str,space,*,gamma=None,lambd=None,mu=None):
    aux_image=space.gray_tone(image)
    edge_image=space.inverse_function(filter(space.function(aux_image,lambd)),lambd)
    if op=='+':
        pre_image=space.sum(aux_image,edge_image,gamma)
    elif op=='-':
        pre_image=space.sub(aux_image,edge_image,gamma)
    else:
        raise ValueError('Invalid operator')
    if mu is not None:
        return space.inverse_gray_tone(pre_image,mu)
    else:
        return space.inverse_gray_tone(pre_image)
    