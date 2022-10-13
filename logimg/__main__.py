from .models.hlip import HLIPImage
from .algorithms.basic_operations import space_mul_img
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from .emee import emee
from skimage import io

a=io.imread('000000037751.jpg')
ha=HLIPImage(a)