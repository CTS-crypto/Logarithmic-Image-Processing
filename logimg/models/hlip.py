import math
import numpy as np
import matplotlib.pyplot as plt
from .logimg import LogImage,LogSpace

class HLIPImage(LogImage):
    def __init__(self,image:np.ndarray,M=256) -> None:
        eps=0.00001
        aux_image=np.maximum(eps,image)
        aux_image=2/M*(aux_image-M/2)
        self.image=np.array( [ [ 0.5 * math.log((1+aux_image[i][j])/(1-aux_image[i][j])) for j in range(image.shape[1])] for i in range(image.shape[0])])
        self.M=M

    def __add__(self,other:'HLIPImage')->'HLIPImage':
        add_image=HLIPImage(np.array([[1]]),self.M)
        if isinstance(other,HLIPImage):
            add_image.image=self.image+other.image
        elif isinstance(other,int) or isinstance(other,float):
            add_image.image=self.image+other
        else:
            raise TypeError('Invalid argument for sum')
        return add_image

    def __sub__(self,other:'HLIPImage')->'HLIPImage':
        add_image=HLIPImage(np.array([[1]]),self.M)
        if isinstance(other,HLIPImage):
            add_image.image=self.image-other.image
        elif isinstance(other,int) or isinstance(other,float):
            add_image.image=self.image-other
        else:
            raise TypeError('Invalid argument for sub')
        return add_image
        
    def __mul__(self,other)->'HLIPImage':
        if isinstance(other,int) or isinstance(other,float):         
            mul_image=HLIPImage(np.array([[1]]),self.M)
            mul_image.image=other*self.image
            return mul_image
        elif isinstance(other,HLIPImage):
            mul_image=HLIPImage(np.array([[1]]),self.M)
            mul_image.image=self.image*other.image
            return mul_image
        raise TypeError('Invalid argument for multiplication')

    def transform(self)->np.ndarray:
        aux= np.array( [ [ (math.e**(2*self.image[i][j])-1)/((math.e**(2*self.image[i][j])+1)) for j in range(self.image.shape[1])] for i in range(self.image.shape[0])])
        return self.M/2*(aux+1)

class HLIPSpace(LogSpace):
    def __init__(self, M=256) -> None:
        super().__init__(M)

    def gray_tone(self,f):
        eps=0.00001
        if isinstance(f,np.ndarray):
            f_aux=np.maximum(eps,f)
        else:
            f_aux=max(eps,f)
        return 2/self.M*(f_aux-self.M/2)

    def inverse_gray_tone(self,f):
        return self.M/2*(f+1)

    def function(self, f):
        if isinstance(f,np.ndarray):
            return np.array( [ [ 0.5 * math.log((1+f[i][j])/(1-f[i][j])) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return 0.5 * math.log((1+f)/(1-f))

    def inverse_function(self, f):
        if isinstance(f,np.ndarray):
            return np.array( [ [ (math.e**(2*f[i][j])-1)/((math.e**(2*f[i][j])+1)) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return (math.e**(2*f)-1)/(math.e**(2*f)+1)

    def sum(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist())
        else:
            g_aux=g
        return (f_aux+g_aux)/(1+f_aux*g_aux)

    def sub(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist())
        else:
            g_aux=g
        return (f_aux-g_aux)/(1-f_aux*g_aux)

    def mul(self,f,g):
        f_aux=self.function(f)
        g_aux=self.function(g)
        return self.inverse_function(f_aux*g_aux)

    def s_mul(self,f,scalar):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return ((1+f_aux)**scalar-(1-f_aux)**scalar)/((1+f_aux)**scalar+(1-f_aux)**scalar)

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