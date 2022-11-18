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
