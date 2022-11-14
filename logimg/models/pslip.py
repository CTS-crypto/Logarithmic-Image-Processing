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

    def function(self, f):
        return f/(1-f)

    def inverse_function(self, f):
        return f/(1+f)

    def sum(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist())
        else:
            g_aux=g
        return (f_aux+g_aux-2*f_aux*g_aux)/(1-f_aux*g_aux)

    def sub(self,f,g):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist())
        else:
            g_aux=g
        return (f_aux-g_aux)/(1+f_aux*g_aux-2*g_aux)

    def s_mul(self,f,scalar):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return scalar*f_aux/(1+(scalar-1)*f_aux)

    def show_curve(self):
        x=[self.gray_tone(i/256*self.M) for i in range(256)]
        x.append(self.gray_tone(255.99/256*self.M))
        plt.plot(x, [self.function(i) for i in x])
        p1=(self.gray_tone(0),self.function(self.gray_tone(1)))
        p2=(self.gray_tone(128/256*self.M),self.function(self.gray_tone(128/256*self.M)))
        p3=(self.gray_tone(255/256*self.M),self.function(self.gray_tone(255/256*self.M)))
        plt.plot(p1[0],p1[1],marker='o',color='red',label=p1)
        plt.plot(p2[0],p2[1],marker='o',color='orange',label=p2)
        plt.plot(p3[0],p3[1],marker='o',color='yellow',label=p3)
        plt.legend()
        plt.title(f'Curva representativa del isomorfismo PSLIP, M = {self.M}')
        plt.xlim(-0.1,1.1)
        plt.ylim(-5,300)
        plt.xlabel('Escala de gris')
        plt.ylabel('φ')
        plt.grid()
        plt.show()