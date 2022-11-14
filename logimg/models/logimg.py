import numpy as np
import matplotlib.pyplot as plt
from abc import ABC,abstractmethod

class LogImage(ABC):
    @abstractmethod
    def __init__(self,image:np.ndarray,M=256) -> None:
        self.image
        self.M

    @property
    def shape(self)->tuple:
        return self.image.shape
    
    def __repr__(self) -> str:
        return self.image.__repr__()

    def __array__(self,dtype=None):
        return self.image

    def __getitem__(self,index):
        return self.image[index]

    @abstractmethod
    def __add__(self,other)->'LogImage':
        ...

    @abstractmethod
    def __sub__(self,other)->'LogImage':
        ...

    @abstractmethod          
    def __mul__(self,other)->'LogImage':
        ...

    def __pow__(self,scalar):
        if not (isinstance(scalar,int) or isinstance(scalar,float)):
            raise Exception("Invalid scalar type")
        new_image = type(self)(np.zeros((1,1),self.M))
        new_image.image = self.image**scalar
        return new_image

    @abstractmethod
    def transform(self)->np.ndarray:
        ...

    def show(self)->None:
        plt.imshow(self.image, cmap='gray', interpolation='nearest') 
        plt.show()

    def histogram(self)->None:
        plt.hist(self.image)
        plt.show()

class LogSpace(ABC):
    @abstractmethod
    def __init__(self,M=256) -> None:
        self.M=M

    @abstractmethod
    def gray_tone(self,f):
        ...

    @abstractmethod
    def inverse_gray_tone(self,g):
        ...

    @abstractmethod
    def function(self,f):
        ...

    @abstractmethod
    def inverse_function(self,f):
        ...

    @abstractmethod
    def sum(self,f,g):
        ...

    @abstractmethod
    def sub(self,f,g):
        ...

    def mul(self,f,g):
        f_aux=self.function(f)
        g_aux=self.function(g)
        return self.inverse_function(f_aux*g_aux)

    @abstractmethod
    def s_mul(self,f,scalar):
        ...

    @abstractmethod
    def show_curve(self):
        ...