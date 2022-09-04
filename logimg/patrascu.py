import math
import numpy as np
import matplotlib.pyplot as plt
from logimg import LogImage,LogSpace

class PatrascuImage(LogImage):
    def __init__(self,image:np.ndarray,M=256) -> None:
        aux_image=(np.array(image.tolist())+1)/(M+1)
        self.image=np.array( [ [ M*0.25*math.log(aux_image[i][j]/(1-aux_image[i][j])) for j in range(aux_image.shape[1])] for i in range(aux_image.shape[0])])
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
        return np.array( [ [ (self.M+1)*math.e**(4*self.image[i][j]/self.M)/(1+math.e**(4*self.image[i][j]/self.M))-1 for j in range(self.image.shape[1])] for i in range(self.image.shape[0])])
        

class PatrascuSpace(LogSpace):
    def __init__(self, M=256) -> None:
        super().__init__(M)
        
    def sum(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=(np.array(f.tolist())+1)/(self.M+1)
        else:
            f_aux=(f+1)/(self.M+1)
        if isinstance(g,np.ndarray):
            g_aux=(np.array(g.tolist())+1)/(self.M+1)
        else:
            g_aux=(g+1)/(self.M+1)
        return (self.M+1)*(f_aux*g_aux)/((1-f_aux)*(1-g_aux)+f_aux*g_aux)-1


    def sub(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=(np.array(f.tolist())+1)/(self.M+1)
        else:
            f_aux=(f+1)/(self.M+1)
        if isinstance(g,np.ndarray):
            g_aux=(np.array(g.tolist())+1)/(self.M+1)
        else:
            g_aux=(g+1)/(self.M+1)
        return (self.M+1)*(f_aux*(1-g_aux))/((1-f_aux)*g_aux+(1-g_aux)*f_aux)-1

    def mul(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=PatrascuImage(f,self.M)
        else:
            f_aux=PatrascuImage([[f]],self.M)
        if isinstance(g,np.ndarray):
            g_aux=PatrascuImage(g,self.M)
        else:
            g_aux=PatrascuImage([[g]],self.M)
        if isinstance(f,np.ndarray):
            return (f_aux*g_aux).transform()
        else:
            return (f_aux*g_aux).transform()[0][0]

    def s_mul(self,f,scalar):
        if isinstance(f,np.ndarray):
            f_aux=(np.array(f.tolist())+1)/(self.M+1)
        else:
            f_aux=(f+1)/(self.M+1)
        return (self.M+1)*f_aux**scalar/(f_aux**scalar+(1-f_aux)**scalar)-1

    def show_curve(self):
        x=range(256)
        plt.plot(x, [ self.M*0.25*math.log(((i+1)/(self.M+1))/(1-(i+1)/(self.M+1))) for i in x])
        plt.title('Curva representativa logarítmica del isomorfismo φ')
        plt.xlim(-50,300)
        plt.ylim(-400,400)
        plt.xlabel('gray-scale')
        plt.ylabel('φ')
        plt.grid()
        plt.show()