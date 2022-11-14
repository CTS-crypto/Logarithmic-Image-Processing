from .logimg import LogSpace
import numpy as np
import matplotlib.pyplot as plt

class PPSLIPSpace(LogSpace):
    def __init__(self, M=256,gamma=256,k=256,lambd=256) -> None:
        super().__init__(M)
        self.gamma=gamma
        self.k=k
        self.lambd=lambd

    def gray_tone(self,f):
        if isinstance(f,np.ndarray):
            return np.array(f.tolist())
        else:
            return f

    def inverse_gray_tone(self,f):
        return self.gray_tone(f)

    def function(self, f, lambd=None):
        if lambd is None:
            lambd=self.lambd
        return f/(lambd-f)

    def inverse_function(self, f, lambd=None):
        if lambd is None:
            lambd=self.lambd
        return lambd*f/(1+f)

    def sum(self,f,g,gamma=None):
        if gamma is None:
            gamma=self.gamma
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist())
        else:
            g_aux=g
        return (f_aux+g_aux-2*f_aux*g_aux/gamma)/(1-f_aux*g_aux/gamma**2)

    def sub(self,f,g,k=None):
        if k is None:
            k=self.k
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist())
        else:
            g_aux=g
        return (f_aux-g_aux)/(1+f_aux*g_aux/k**2-2*g_aux/k)

    def mul(self,f,g,lambd=None):
        f_aux=self.function(f,lambd)
        g_aux=self.function(g,lambd)
        return self.inverse_function(f_aux*g_aux,lambd)

    def s_mul(self,f,scalar,gamma=None):
        if gamma is None:
            gamma=self.gamma
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return scalar*f_aux/(1+(scalar-1)*f_aux/gamma)

    def show_curve(self,lambd=None):
        if lambd is None:
            lambd=self.lambd
        x=[i/256*self.M for i in range(int(256/self.M*lambd))]
        x.append(255.9999/256*lambd)
        plt.plot(x, [self.function(i,lambd) for i in x])
        p1=(0,self.function(0,lambd))
        p2=(128/256*self.M,self.function(128/256*self.M,lambd))
        p3=(255/256*self.M,self.function(255/256*self.M,lambd))
        plt.plot(p1[0],p1[1],marker='o',color='red',label=p1)
        plt.plot(p2[0],p2[1],marker='o',color='orange',label=p2)
        plt.plot(p3[0],p3[1],marker='o',color='yellow',label=p3)
        plt.legend()
        plt.title(f'Curva representativa del isomorfismo PPSLIP, M = {self.M}, λ(M) = {lambd}')
        plt.xlim(-5*lambd/256, lambd+5*lambd/256)
        plt.ylim(-5*lambd/self.M, 256/self.M*lambd+10*lambd/self.M)
        plt.xlabel('Escala de gris')
        plt.ylabel('φ')
        plt.grid()
        plt.show()