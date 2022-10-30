import numpy as np
import matplotlib.pyplot as plt
from .logimg import LogImage,LogSpace

class PSLIPImage(LogImage):
    def __init__(self,image:np.ndarray,M=256) -> None:
        aux_image=np.array(image.tolist())
        self.image=aux_image/(M-aux_image)
        self.M=M

    def __add__(self,other:'PSLIPImage')->'PSLIPImage':
        add_image=PSLIPImage(np.array([[1]]),self.M)
        if isinstance(other,PSLIPImage):
            add_image.image=self.image+other.image
        elif isinstance(other,int) or isinstance(other,float):
            add_image.image=self.image+other
        else:
            raise TypeError('Invalid argument for sum')
        return add_image

    def __sub__(self,other:'PSLIPImage')->'PSLIPImage':
        add_image=PSLIPImage(np.array([[1]]),self.M)
        if isinstance(other,PSLIPImage):
            add_image.image=self.image-other.image
        elif isinstance(other,int) or isinstance(other,float):
            add_image.image=self.image-other
        else:
            raise TypeError('Invalid argument for sub')
        return add_image
        
    def __mul__(self,other)->'PSLIPImage':
        if isinstance(other,int) or isinstance(other,float):         
            mul_image=PSLIPImage(np.array([[1]]),self.M)
            mul_image.image=other*self.image
            return mul_image
        elif isinstance(other,PSLIPImage):
            mul_image=PSLIPImage(np.array([[1]]),self.M)
            mul_image.image=self.image*other.image
            return mul_image
        raise TypeError('Invalid argument for multiplication')

    def transform(self)->np.ndarray:
        return self.M*self.image/(1+self.image)
        

class PSLIPSpace(LogSpace):
    def __init__(self, M=256) -> None:
        super().__init__(M)

    def gray_tone(self,f):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return f_aux/self.M

    def inverse_gray_tone(self,f):
        return self.M*f

    def equation(self, f):
        return f/(1-f)

    def inverse_equation(self, f):
        return f/(1+f)

    def sum(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())/self.M
        else:
            f_aux=f/self.M
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist())/self.M
        else:
            g_aux=g/self.M
        return self.M*(f_aux+g_aux-2*f_aux*g_aux)/(1-f_aux*g_aux)

    def sub(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())/self.M
        else:
            f_aux=f/self.M
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist())/self.M
        else:
            g_aux=g/self.M
        return self.M*(f_aux-g_aux)/(1+f_aux*g_aux-2*g_aux)

    def mul(self,f,g):
        f_aux=self.equation(f)
        g_aux=self.equation(g)
        return self.inverse_equation(f_aux*g_aux)

    def s_mul(self,f,scalar):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())/self.M
        else:
            f_aux=f/self.M
        return self.M*scalar*f_aux/(1+(scalar-1)*f_aux)

    def show_curve(self):
        x=range(256)
        plt.plot(x, [i/(1-i/self.M) for i in x])
        plt.title('Curva representativa logarítmica del isomorfismo φ')
        plt.xlim(0,300)
        plt.ylim(0,10000)
        plt.xlabel('gray-scale')
        plt.ylabel('φ')
        plt.grid()
        plt.show()