import math
import numpy as np
import matplotlib.pyplot as plt
from .logimg import LogImage,LogSpace

class LIPImage(LogImage):
    def __init__(self,image:np.ndarray,M=256) -> None:
        eps=0.00001
        aux_image=np.maximum(eps,image)
        self.image=np.array( [ [ -M * math.log(aux_image[i][j]/M) for j in range(image.shape[1])] for i in range(image.shape[0])])
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
        eps=0.00001
        aux_image=np.maximum(eps,self.image)
        return np.array( [ [ self.M/math.e**(aux_image[i][j]/self.M) for j in range(self.image.shape[1])] for i in range(self.image.shape[0])])
        
class LIPSpace(LogSpace):
    def __init__(self, M=256) -> None:
        super().__init__(M)

    def gray_tone(self,f):
        eps=0.00001
        if isinstance(f,np.ndarray):
            f_aux=np.maximum(eps,f)
        else:
            f_aux=max(eps,f)
        return self.M-f_aux

    def inverse_gray_tone(self,f):
        return self.gray_tone(f)

    def function(self, f):
        if isinstance(f,np.ndarray):
            return np.array( [ [ -self.M * math.log(1-f[i][j]/self.M) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return -self.M * math.log(1-f/self.M)

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
        return (f_aux-g_aux)/(1-g_aux/self.M)

    def s_mul(self,f,scalar):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return self.M-self.M*(1-f_aux/self.M)**scalar

    def show_curve(self):
        x=[i/256*self.M for i in range(-300,256)]
        x.append(255.9999/256*self.M)
        plt.plot(x, [self.function(i) for i in x])
        p1=(0,self.function(0))
        p2=(128/256*self.M,self.function(128/256*self.M))
        p3=(255/256*self.M,self.function(255/256*self.M))
        plt.plot(p1[0],p1[1],marker='o',color='red',label=p1)
        plt.plot(p2[0],p2[1],marker='o',color='orange',label=p2)
        plt.plot(p3[0],p3[1],marker='o',color='yellow',label=p3)
        plt.legend()
        plt.title(f'Curva representativa del isomorfismo LIP, M = {self.M}')
        plt.xlim(-300/256*self.M,300/256*self.M)
        plt.ylim(-500/256*self.M,2000/256*self.M)
        plt.xlabel('Escala de gris')
        plt.ylabel('φ')
        plt.grid()
        plt.show()