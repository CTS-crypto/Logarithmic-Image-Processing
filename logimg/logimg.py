from hashlib import new
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

    @property
    def shape(self)->tuple:
        return self.image.shape

    def show(self)->None:
        plt.imshow(self.image, cmap='gray', interpolation='nearest') 
        plt.show()

    def histogram(self)->None:
        plt.hist(self.image)
        plt.show()

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

class LogSpace(ABC):
    @abstractmethod
    def __init__(self,M=256) -> None:
        self.M=M

    def neg(self,f):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return self.M-1-f_aux

    @abstractmethod
    def equation(self,f):
        ...

    @abstractmethod
    def inverse_equation(self,f):
        ...

    @abstractmethod
    def sum(self,f,g):
        ...

    @abstractmethod
    def sub(self,f,g):
        ...

    @abstractmethod
    def mul(self,f,g):
        ...

    @abstractmethod
    def s_mul(self,f,scalar):
        ...

    @abstractmethod
    def show_curve(self):
        ...