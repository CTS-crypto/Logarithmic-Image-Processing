import math
import numpy as np
import matplotlib.pyplot as plt
from .logimg import LogImage,LogSpace

class SLIPImage(LogImage):
    def __init__(self,image:np.ndarray,M=256) -> None:
        self.image=np.array( [ [ -M*np.sign(image[i][j])*math.log(0.0001/M if image[i][j]==0 else math.abs(image[i][j])/M) for j in range(image.shape[1])] for i in range(image.shape[0])])
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
        
    def __mul__(self,other)->'SLIPImage':
        if isinstance(other,int) or isinstance(other,float):         
            mul_image=SLIPImage(np.array([[1]]),self.M)
            mul_image.image=other*self.image
            return mul_image
        elif isinstance(other,SLIPImage):
            mul_image=SLIPImage(np.array([[1]]),self.M)
            mul_image.image=self.image*other.image
            return mul_image
        raise TypeError('Invalid argument for multiplication')

    def transform(self)->np.ndarray:
        return np.array( [ [ self.M*np.sign(self.image[i][j])*(1-math.e**(-abs(self.image[i][j])/self.M)) for j in range(self.image.shape[1])] for i in range(self.image.shape[0])])
        

class SLIPSpace(LogSpace):
    def __init__(self, M=256) -> None:
        super().__init__(M)

    def equation(self, f):
        if isinstance(f,np.ndarray):
            return np.array( [ [ -self.M * np.sign(f[i][j]) * math.log(1.0001-math.abs(f[i][j])/self.M if math.abs(f[i][j])==math.abs(self.M) else 1-math.abs(f[i][j])/self.M) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return -self.M * np.sign(f) * math.log(1.0001-math.abs(f)/self.M if math.abs(f)==math.abs(self.M) else 1-math.abs(f)/self.M)

    def inverse_equation(self, f):
        if isinstance(f,np.ndarray):
            return np.array( [ [ self.M*np.sign(f[i][j])*(1-1/math.e**(math.abs(f[i][j])/self.M)) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return self.M*np.sign(f)*(1-1/math.e**(math.abs(f)/self.M))
            
    def sum(self,f,g):
        def calculate_pixel_sum(x,y):
            gamma1=np.sign(x)*np.sign(x+y)
            gamma2=np.sign(y)*np.sign(x+y)
            return self.M*np.sign(x+y)*(1-(1-abs(x)/self.M)**gamma1*(1-abs(y)/self.M)**gamma2)
        if isinstance(f,np.ndarray):
            f_aux=f.tolist()
        if isinstance(g,np.ndarray):
            g_aux=g.tolist()
        if isinstance(f,np.ndarray) and isinstance(g,np.ndarray):
            return np.array( [ [ calculate_pixel_sum(f_aux[i][j],g_aux[i][j]) for j in range(len(f_aux[0]))] for i in range(len(f_aux))])
        elif isinstance(f,np.ndarray):
            return np.array( [ [ calculate_pixel_sum(f_aux[i][j],g) for j in range(len(f_aux[0]))] for i in range(len(f_aux))])
        elif isinstance(g,np.ndarray):
            return np.array( [ [ calculate_pixel_sum(f,g_aux[i][j]) for j in range(len(f_aux[0]))] for i in range(len(f_aux))])
        else:
            return calculate_pixel_sum(f,g)

    def sub(self,f,g):
        return self.sum(f,self.s_mul(g,-1))

    def mul(self,f,g):
        f_aux=self.equation(f)
        g_aux=self.equation(g)
        return self.inverse_equation(f_aux*g_aux)

    def s_mul(self,f,scalar):
        def calculate_pixel_s_mul(x,scalar):
            return self.M*np.sign(x*scalar)*(1-(1-abs(x)/self.M)**abs(scalar))
        if isinstance(f_aux,np.ndarray):
            f_aux=f.tolist()
            return np.array( [ [ calculate_pixel_s_mul(f_aux[i][j],scalar) for j in range(len(f_aux[0]))] for i in range(len(f_aux))])
        else:
            return calculate_pixel_s_mul(f,scalar)

    def show_curve(self):
        x=range(-256,257)
        plt.plot(x, [-self.M*np.sign(i)*math.log(1-abs(i)/self.M) for i in x])
        plt.title('Curva representativa logarítmica del isomorfismo φ')
        plt.xlim(-300,300)
        plt.ylim(-1600,1600)
        plt.xlabel('gray-scale')
        plt.ylabel('φ')
        plt.grid()
        plt.show()