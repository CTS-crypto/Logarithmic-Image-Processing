import math
import numpy as np
import matplotlib.pyplot as plt
from .logimg import LogImage,LogSpace

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

    def gray_tone(self,f):
        if isinstance(f,np.ndarray):
            return np.array(f.tolist())
        else:
            return f

    def inverse_gray_tone(self,f):
        return self.gray_tone(f)

    def function(self, f):
        if isinstance(f,np.ndarray):
            return np.array( [ [ -self.M * np.sign(f[i][j]) * math.log(1-abs(f[i][j])/self.M) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return -self.M * np.sign(f) * math.log(1-abs(f)/self.M)

    def inverse_function(self, f):
        if isinstance(f,np.ndarray):
            return np.array( [ [ self.M*np.sign(f[i][j])*(1-1/math.e**(abs(f[i][j])/self.M)) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return self.M*np.sign(f)*(1-1/math.e**(abs(f)/self.M))
            
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

    def s_mul(self,f,scalar):
        def calculate_pixel_s_mul(x,scalar):
            return self.M*np.sign(x*scalar)*(1-(1-abs(x)/self.M)**abs(scalar))
        if isinstance(f,np.ndarray):
            f_aux=f.tolist()
            return np.array( [ [ calculate_pixel_s_mul(f_aux[i][j],scalar) for j in range(len(f_aux[0]))] for i in range(len(f_aux))])
        else:
            return calculate_pixel_s_mul(f,scalar)

    def show_curve(self):
        x=[i/256*self.M for i in range(-255,255)]
        x.insert(0,-255.9999/256*self.M)
        x.append(255.9999/256*self.M)
        plt.plot(x, [self.function(i) for i in x])
        p1=(0,self.function(0))
        p2=(128/256*self.M,self.function(128/256*self.M))
        p3=(255/256*self.M,self.function(255/256*self.M))
        p4=(-128/256*self.M,self.function(-128/256*self.M))
        p5=(-255/256*self.M,self.function(-255/256*self.M))
        plt.plot(p1[0],p1[1],marker='o',color='red',label=p1)
        plt.plot(p2[0],p2[1],marker='o',color='orange',label=p2)
        plt.plot(p3[0],p3[1],marker='o',color='yellow',label=p3)
        plt.plot(p4[0],p4[1],marker='o',color='green',label=p4)
        plt.plot(p5[0],p5[1],marker='o',color='purple',label=p5)
        plt.legend()
        plt.title(f'Curva representativa del isomorfismo SLIP, M = {self.M}')
        plt.xlim(-300/256*self.M,300/256*self.M)
        plt.ylim(-2000/256*self.M,2000/256*self.M)
        plt.xlabel('Escala de gris')
        plt.ylabel('φ')
        plt.grid()
        plt.show()