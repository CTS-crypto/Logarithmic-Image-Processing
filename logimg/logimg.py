import numpy as np
import matplotlib.pyplot as plt
from abc import ABC,abstractmethod

class LogImage(ABC):
    @abstractmethod
    def __init__(self,image:np.ndarray,M=256) -> None:
        self.image
        self.M

    def __repr__(self) -> str:
        return self.image.__repr__()

    def __array__(self,dtype=None):
        return self.image

    def __getitem__(self,index):
        return self.image[index]

    def show(self)->None:
        plt.imshow(self.image, cmap='gray', interpolation='nearest') 
        plt.show()

    def histogram(self)->None:
        plt.hist(self.image)
        plt.show()

    @abstractmethod
    def __add__(self,other:'LogImage')->'LogImage':
        ...

    @abstractmethod
    def __sub__(self,other:'LogImage')->'LogImage':
        ...

    @abstractmethod          
    def __mul__(self,scalar)->'LogImage':
        ...

    @abstractmethod
    def transform(self)->np.ndarray:
        ...

class LogSpace(ABC):
    @abstractmethod
    def __init__(self,M=256) -> None:
        self.M=M

    @abstractmethod
    def sum(self,f:np.ndarray,g:np.ndarray)->np.ndarray:
        ...

    @abstractmethod
    def sub(self,f:np.ndarray,g:np.ndarray)->np.ndarray:
        ...

    @abstractmethod
    def s_mul(self,f:np.ndarray,scalar)->np.ndarray:
        ...

    @abstractmethod
    def show_curve(self):
        ...