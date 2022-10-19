import math
import numpy as np
import matplotlib.pyplot as plt
from .logimg import LogImage,LogSpace

class HLIPImage(LogImage):
    def __init__(self,image:np.ndarray,M=256) -> None:
        aux_image=2/M*(image-M/2)
        zero_replace=(1-0.9999)/(1+0.9999)
        self.image=np.array( [ [ M/2 * math.log( zero_replace if aux_image[i][j]==-1 else (1+aux_image[i][j])/(1-aux_image[i][j])) for j in range(image.shape[1])] for i in range(image.shape[0])])
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
        aux= np.array( [ [ (math.e**(2/self.M*self.image[i][j])-1)/((math.e**(2/self.M*self.image[i][j])+1)) for j in range(self.image.shape[1])] for i in range(self.image.shape[0])])
        return self.M/2*(aux+1)

class HLIPSpace(LogSpace):
    def __init__(self, M=256) -> None:
        super().__init__(M)

    def equation(self, f):
        if isinstance(f,np.ndarray):
            zero_replace=(1-0.9999)/(1+0.9999)
            return np.array( [ [ self.M/2 * math.log( zero_replace if f[i][j]==-1 else (1-f[i][j])/(1+f[i][j])) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return self.M/2 * math.log( zero_replace if f==-1 else (1-f)/(1+f))

    def inverse_equation(self, f):
        if isinstance(f,np.ndarray):
            return np.array( [ [ (math.e**(2/self.M*f[i][j])-1)/((math.e**(2/self.M*f[i][j])+1)) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return (math.e**(2/self.M*f)-1)/((math.e**(2/self.M*f)+1))
     
    def gray_tone(self,f):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return 2/self.M*(f_aux-self.M/2)

    def inverse_gray_tone(self,f):
        return self.M/2*(f+1)

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
        f_aux=self.equation(f)
        g_aux=self.equation(g)
        return self.inverse_equation(f_aux*g_aux)

    def s_mul(self,f,scalar):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return ((1+f)**scalar-(1-f)**scalar)/((1+f)**scalar+(1-f)**scalar)

    def show_curve(self):
        x=range(257)
        plt.plot(x, [self.equation(i) for i in x])
        plt.title('Curva representativa logarítmica del isomorfismo φ')
        plt.xlim(0,300)
        plt.ylim(0,1600)
        plt.xlabel('gray-scale')
        plt.ylabel('φ')
        plt.grid()
        plt.show()