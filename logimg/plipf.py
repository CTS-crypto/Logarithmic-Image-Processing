import numpy as np
import matplotlib.pyplot as plt
from logimg import LogImage,LogSpace

class PLIPFImage(LogImage):
    def __init__(self,image:np.ndarray,M=256) -> None:
        aux_image=np.array(image.tolist())
        self.image=M*aux_image/(M-aux_image)
        self.M=M

    def __add__(self,other:'PLIPFImage')->'PLIPFImage':
        add_image=PLIPFImage(np.array([[1]]),self.M)
        if isinstance(other,PLIPFImage):
            add_image.image=self.image+other.image
        elif isinstance(other,int) or isinstance(other,float):
            add_image.image=self.image+other
        else:
            raise TypeError('Invalid argument for sum')
        return add_image

    def __sub__(self,other:'PLIPFImage')->'PLIPFImage':
        add_image=PLIPFImage(np.array([[1]]),self.M)
        if isinstance(other,PLIPFImage):
            add_image.image=self.image-other.image
        elif isinstance(other,int) or isinstance(other,float):
            add_image.image=self.image-other
        else:
            raise TypeError('Invalid argument for sub')
        return add_image
        
    def __mul__(self,scalar)->'PLIPFImage':
        if not (isinstance(scalar,int) or isinstance(scalar,float)):
            raise TypeError('Invalid argument for multiplication') 
        add_image=PLIPFImage(np.array([[1]]),self.M)
        add_image.image=scalar*self.image
        return add_image

    def transform(self)->np.ndarray:
        return self.M*self.image/(self.M+self.image)
        

class PLIPFSpace(LogSpace):
    def __init__(self, M=256) -> None:
        super().__init__(M)
        
    def sum(self,f:np.ndarray,g:np.ndarray)->np.ndarray:
        f_aux=np.array(f.tolist())/self.M
        g_aux=np.array(g.tolist())/self.M
        return self.M*(f_aux+g_aux-2*f_aux*g_aux)/(1-f_aux*g_aux)

    def sub(self,f:np.ndarray,g:np.ndarray)->np.ndarray:
        f_aux=np.array(f.tolist())/self.M
        g_aux=np.array(g.tolist())/self.M
        return self.M*(f_aux-g_aux)/(1+f_aux*g_aux-2*g_aux)

    def s_mul(self,f:np.ndarray,scalar)->np.ndarray:
        f_aux=np.array(f.tolist())/self.M
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