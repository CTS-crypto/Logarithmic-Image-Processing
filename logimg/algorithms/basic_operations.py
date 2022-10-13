import numpy as np
import matplotlib.pyplot as plt
from ..models.logimg import LogSpace

def sum_img(f,g):
    f_aux=np.array(f.tolist())
    g_aux=np.array(g.tolist())
    return f_aux+g_aux

def space_sum_img(f,g,space:LogSpace):
    f_neg=space.M-f
    g_neg=space.M-g
    return space.neg(space.sum(f_neg,g_neg))

def sub_img(f,g):
    f_aux=np.array(f.tolist())
    g_aux=np.array(g.tolist())
    return f_aux-g_aux

def space_sub_img(f,g,space:LogSpace):
    f_neg=space.M-f
    g_neg=space.M-g
    return space.neg(space.sub(f_neg,g_neg))

def mul_img(f,g):
    f_aux=np.array(f.tolist())
    g_aux=np.array(g.tolist())
    return f_aux*g_aux

def space_mul_img(f,g,space:LogSpace):
    f_neg=space.M-f
    g_neg=space.M-g
    return space.neg(space.mul(f_neg,g_neg))

def s_mul_img(f,g):
    f_aux=np.array(f.tolist())
    return f_aux*g

def space_s_mul_img(f,g,space:LogSpace):
    f_neg=space.M-f
    return space.neg(space.s_mul(f_neg,g))