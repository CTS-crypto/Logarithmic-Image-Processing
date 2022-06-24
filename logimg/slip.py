import math
import numpy as np
import matplotlib.pyplot as plt
from logimg import LogImage,LogSpace

class SLIPImage(LogImage):
    def __init__(self,image:np.ndarray,M=256) -> None:
        self.image=np.array( [ [ -M*np.sign(image[i][j])*math.log(1-abs(image[i][j])/M) for j in range(image.shape[1])] for i in range(image.shape[0])])
        self.M=M

    def __add__(self,other:'SLIPImage')->'SLIPImage':
        add_image=SLIPImage(np.array([[1]]),self.M)
        if isinstance(other,SLIPImage):
            add_image.image=self.image+other.image
        elif isinstance(other,int) or isinstance(other,float):
            add_image.image=self.image+other
        else:
            raise TypeError('Invalid argument for sum')
        return add_image

    def __sub__(self,other:'SLIPImage')->'SLIPImage':
        add_image=SLIPImage(np.array([[1]]),self.M)
        if isinstance(other,SLIPImage):
            add_image.image=self.image-other.image
        elif isinstance(other,int) or isinstance(other,float):
            add_image.image=self.image-other
        else:
            raise TypeError('Invalid argument for sub')
        return add_image
        
    def __mul__(self,scalar)->'SLIPImage':
        if not (isinstance(scalar,int) or isinstance(scalar,float)):
            raise TypeError('Invalid argument for multiplication') 
        add_image=SLIPImage(np.array([[1]]),self.M)
        add_image.image=scalar*self.image
        return add_image

    def transform(self)->np.ndarray:
        return np.array( [ [ self.M*np.sign(self.image[i][j])*(1-math.e**(-abs(self.image[i][j])/self.M)) for j in range(self.image.shape[1])] for i in range(self.image.shape[0])])
        

class SLIPSpace(LogSpace):
    def __init__(self, M=256) -> None:
        super().__init__(M)
        
    def sum(self,f:np.ndarray,g:np.ndarray)->np.ndarray:
        f_aux=f.tolist()
        g_aux=g.tolist()
        def calculate_pixel_sum(x,y):
            gamma1=np.sign(x)*np.sign(x+y)
            gamma2=np.sign(y)*np.sign(x+y)
            return self.M*np.sign(x+y)*(1-(1-abs(x)/self.M)**gamma1*(1-abs(y)/self.M)**gamma2)
        return np.array( [ [ calculate_pixel_sum(f_aux[i][j],g_aux[i][j]) for j in range(len(f_aux[0]))] for i in range(len(f_aux))])

    def sub(self,f:np.ndarray,g:np.ndarray)->np.ndarray:
        return self.sum(f,self.s_mul(g,-1))

    def s_mul(self,f:np.ndarray,scalar)->np.ndarray:
        f_aux=f.tolist()
        def calculate_pixel_s_mul(x,scalar):
            return self.M*np.sign(x*scalar)*(1-(1-abs(x)/self.M)**abs(scalar))
        return np.array( [ [ calculate_pixel_s_mul(f_aux[i][j],scalar) for j in range(len(f_aux[0]))] for i in range(len(f_aux))])

    def show_curve(self):
        x=range(-255,256)
        plt.plot(x, [-self.M*np.sign(i)*math.log(1-abs(i)/self.M) for i in x])
        plt.title('Curva representativa logarítmica del isomorfismo φ')
        plt.xlim(-300,300)
        plt.ylim(-1600,1600)
        plt.xlabel('gray-scale')
        plt.ylabel('φ')
        plt.grid()
        plt.show()