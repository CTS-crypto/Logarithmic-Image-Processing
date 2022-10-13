import math
import numpy as np
import matplotlib.pyplot as plt
from .logimg import LogImage,LogSpace

class PatrascuImage(LogImage):
    def __init__(self,image:np.ndarray,M=256) -> None:
        aux_image=np.array(image.tolist()+0.0001)/M
        self.image=np.array( [ [ 0.25*math.log(aux_image[i][j]/(1-aux_image[i][j])) for j in range(aux_image.shape[1])] for i in range(aux_image.shape[0])])
        self.M=M

    def __add__(self,other:'PatrascuImage')->'PatrascuImage':
        add_image=PatrascuImage(np.array([[1]]),self.M)
        if isinstance(other,PatrascuImage):
            add_image.image=self.image+other.image
        elif isinstance(other,int) or isinstance(other,float):
            add_image.image=self.image+other
        else:
            raise TypeError('Invalid argument for sum')
        return add_image

    def __sub__(self,other:'PatrascuImage')->'PatrascuImage':
        add_image=PatrascuImage(np.array([[1]]),self.M)
        if isinstance(other,PatrascuImage):
            add_image.image=self.image-other.image
        elif isinstance(other,int) or isinstance(other,float):
            add_image.image=self.image-other
        else:
            raise TypeError('Invalid argument for sum')
        return add_image
        
    def __mul__(self,other)->'PatrascuImage':
        if isinstance(other,int) or isinstance(other,float):         
            mul_image=PatrascuImage(np.array([[1]]),self.M)
            mul_image.image=other*self.image
            return mul_image
        elif isinstance(other,PatrascuImage):
            mul_image=PatrascuImage(np.array([[1]]),self.M)
            mul_image.image=self.image*other.image
            return mul_image
        raise TypeError('Invalid argument for multiplication')

    def transform(self)->np.ndarray:
        return np.array( [ [ self.M*math.e**(4*self.image[i][j])/(1+math.e**(4*self.image[i][j]))-0.0001 for j in range(self.image.shape[1])] for i in range(self.image.shape[0])])
        

class PatrascuSpace(LogSpace):
    def __init__(self, M=256) -> None:
        super().__init__(M)

    def equation(self, f):
        if isinstance(f,np.ndarray):
            aux_image=np.array(f.tolist()+0.0001)/self.M
            return np.array( [ [ 0.25*math.log(aux_image[i][j]/(1-aux_image[i][j])) for j in range(aux_image.shape[1])] for i in range(aux_image.shape[0])])
        else:
            f_aux=(f+0.0001)/self.M
            return 0.25*math.log(f_aux/(1-f_aux))

    def inverse_equation(self, f):
        if isinstance(f,np.ndarray):
            return np.array( [ [ self.M*math.e**(4*self.image[i][j])/(1+math.e**(4*self.image[i][j]))-0.0001 for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return self.M*math.e**(4*f)/(1+math.e**(4*f))-0.0001
        
    def sum(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=(np.array(f.tolist())+0.0001)/self.M
        else:
            f_aux=(f+0.0001)/self.M
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist()+0.0001)/self.M
        else:
            g_aux=(g+0.0001)/self.M
        return self.M*(f_aux*g_aux)/((1-f_aux)*(1-g_aux)+f_aux*g_aux)

    def sub(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=(np.array(f.tolist())+0.0001)/self.M
        else:
            f_aux=(f+0.0001)/self.M
        if isinstance(g,np.ndarray):
            g_aux=(np.array(g.tolist())+0.0001)/self.M
        else:
            g_aux=(g+0.0001)/self.M
        return self.M*(f_aux*(1-g_aux))/((1-f_aux)*g_aux+(1-g_aux)*f_aux)

    def mul(self,f,g):
        f_aux=self.equation(f)
        g_aux=self.equation(g)
        return self.inverse_equation(f_aux*g_aux)

    def s_mul(self,f,scalar):
        if isinstance(f,np.ndarray):
            f_aux=(np.array(f.tolist())+0.0001)/self.M
        else:
            f_aux=(f+0.0001)/self.M
        return self.M*f_aux**scalar/(f_aux**scalar+(1-f_aux)**scalar)

    def show_curve(self):
        x=range(256)
        plt.plot(x, [self.equation(i) for i in x])
        plt.title('Curva representativa logarítmica del isomorfismo φ')
        plt.xlim(0,300)
        plt.ylim(0,1600)
        plt.xlabel('gray-scale')
        plt.ylabel('φ')
        plt.grid()
        plt.show()