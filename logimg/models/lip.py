import math
import numpy as np
import matplotlib.pyplot as plt
from .logimg import LogImage,LogSpace

class LIPImage(LogImage):
    def __init__(self,image:np.ndarray,M=256) -> None:
        self.image=np.array( [ [ -M * math.log( 0.0001/M if image[i][j]==0 else image[i][j]/M) for j in range(image.shape[1])] for i in range(image.shape[0])])
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
        
    def __mul__(self,other)->'LIPImage':
        if isinstance(other,int) or isinstance(other,float):         
            mul_image=LIPImage(np.array([[1]]),self.M)
            mul_image.image=other*self.image
            return mul_image
        elif isinstance(other,LIPImage):
            mul_image=LIPImage(np.array([[1]]),self.M)
            mul_image.image=self.image*other.image
            return mul_image
        raise TypeError('Invalid argument for multiplication')

    def transform(self)->np.ndarray:
        return np.array( [ [ self.M/math.e**(self.image[i][j]/self.M) for j in range(self.image.shape[1])] for i in range(self.image.shape[0])])
        
class LIPSpace(LogSpace):
    def __init__(self, M=256) -> None:
        super().__init__(M)

    def gray_tone(self,f):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return self.M-f_aux

    def inverse_gray_tone(self,f):
        return self.gray_tone(f)

    def function(self, f):
        if isinstance(f,np.ndarray):
            return np.array( [ [ -self.M * math.log(0.0001 if f[i][j]==self.M else 1-f[i][j]/self.M) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return -self.M * math.log(1.0001-f/self.M if f==self.M else 1-f/self.M)

    def inverse_function(self, f):
        if isinstance(f,np.ndarray):
            return np.array( [ [ self.M*(1-1/math.e**(f[i][j]/self.M)) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return self.M*(1-1/math.e**(f/self.M))

    def sum(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist())
        else:
            g_aux=g
        return f_aux+g_aux-(f_aux*g_aux)/self.M

    def sub(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist())
        else:
            g_aux=g
        return (f_aux-g_aux)/(1+0.0001-g_aux/self.M)

    def mul(self,f,g):
        f_aux=self.function(f)
        g_aux=self.function(g)
        return self.inverse_function(f_aux*g_aux)

    def s_mul(self,f,scalar):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return self.M-self.M*(1-f_aux/self.M)**scalar

    def show_curve(self):
        x=range(257)
        plt.plot(x, [self.function(i) for i in x])
        plt.title('Curva representativa logarítmica del isomorfismo φ')
        plt.xlim(0,300)
        plt.ylim(0,1600)
        plt.xlabel('gray-scale')
        plt.ylabel('φ')
        plt.grid()
        plt.show()