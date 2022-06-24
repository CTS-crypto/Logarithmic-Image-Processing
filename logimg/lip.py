import math
import numpy as np
import matplotlib.pyplot as plt
from logimg import LogImage,LogSpace

class LIPImage(LogImage):
    def __init__(self,image:np.ndarray,M=256) -> None:
        self.image=np.array( [ [ -M * math.log(1-image[i][j]/M) for j in range(image.shape[1])] for i in range(image.shape[0])])
        self.M=M

    def __add__(self,other:'LIPImage')->'LIPImage':
        add_image=LIPImage(np.array([[1]]),self.M)
        if isinstance(other,LIPImage):
            add_image.image=self.image+other.image
        elif isinstance(other,int) or isinstance(other,float):
            add_image.image=self.image+other
        else:
            raise TypeError('Invalid argument for sum')
        return add_image

    def __sub__(self,other:'LIPImage')->'LIPImage':
        add_image=LIPImage(np.array([[1]]),self.M)
        if isinstance(other,LIPImage):
            add_image.image=self.image-other.image
        elif isinstance(other,int) or isinstance(other,float):
            add_image.image=self.image-other
        else:
            raise TypeError('Invalid argument for sub')
        return add_image
        
    def __mul__(self,scalar)->'LIPImage':
        if not (isinstance(scalar,int) or isinstance(scalar,float)):
            raise TypeError('Invalid argument for multiplication') 
        add_image=LIPImage(np.array([[1]]),self.M)
        add_image.image=scalar*self.image
        return add_image

    def transform(self)->np.ndarray:
        return np.array( [ [ self.M*(1-1/math.e**(self.image[i][j]/self.M)) for j in range(self.image.shape[1])] for i in range(self.image.shape[0])])
        

class LIPSpace(LogSpace):
    def __init__(self, M=256) -> None:
        super().__init__(M)
        
    def sum(self,f:np.ndarray,g:np.ndarray)->np.ndarray:
        f_aux=np.array(f.tolist())
        g_aux=np.array(g.tolist())
        return f_aux+g_aux-(f_aux*g_aux)/self.M

    def sub(self,f:np.ndarray,g:np.ndarray)->np.ndarray:
        f_aux=np.array(f.tolist())
        g_aux=np.array(g.tolist())
        return (f_aux-g_aux)/(1-g_aux/self.M)

    def s_mul(self,f:np.ndarray,scalar)->np.ndarray:
        f_aux=np.array(f.tolist())
        return self.M-self.M*(1-f_aux/self.M)**scalar

    def show_curve(self):
        x=range(256)
        plt.plot(x, [-self.M * math.log(1-i/self.M) for i in x])
        plt.title('Curva representativa logarítmica del isomorfismo φ')
        plt.xlim(0,300)
        plt.ylim(0,1600)
        plt.xlabel('gray-scale')
        plt.ylabel('φ')
        plt.grid()
        plt.show()

LIPSpace().show_curve()