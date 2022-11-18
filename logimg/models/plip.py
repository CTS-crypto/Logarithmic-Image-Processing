import math
import numpy as np
import matplotlib.pyplot as plt
from .logimg import LogSpace

class PLIPSpace(LogSpace):
    def __init__(self, M=256,mu=256,gamma=256,k=256,lambd=256,beta=1) -> None:
        super().__init__(M)
        self.mu=mu
        self.gamma=gamma
        self.k=k
        self.lambd=lambd
        self.beta=beta

    def gray_tone(self, f, mu=None):
        if mu is None:
            mu=self.mu
        eps=0.00001
        if isinstance(f,np.ndarray):
            f_aux=np.maximum(eps,f)
        else:
            f_aux=max(eps,f)
        return mu-f_aux

    def inverse_gray_tone(self,f,mu=None):
        if mu is None:
            mu=self.mu
        return mu - f

    def function(self, f, lambd=None, beta=None):
        if lambd is None:
            lambd=self.lambd
        if beta is None:
            beta=self.beta
        if isinstance(f,np.ndarray):
            return np.array( [ [ -lambd * math.log( 1-f[i][j]/lambd)**beta for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return -lambd * math.log(1-f/lambd)**beta

    def inverse_function(self, f, lambd=None, beta=None):
        if lambd is None:
            lambd=self.lambd
        if beta is None:
            beta=self.beta
        if isinstance(f,np.ndarray):
            return np.array( [ [ lambd*(1-1/math.e**(f[i][j]/(lambd*beta))) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return lambd*(1-1/math.e**(f/(lambd*beta)))
     
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
        return f_aux+g_aux-(f_aux*g_aux)/gamma

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
        return (f_aux-g_aux)/(1-g_aux/k)

    def mul(self,f,g,lambd=None,beta=None):
        f_aux=self.function(f,lambd,beta)
        g_aux=self.function(g,lambd,beta)
        return self.inverse_function(f_aux*g_aux,lambd,beta)

    def s_mul(self,f,scalar,gamma=None):
        if gamma is None:
            gamma=self.gamma
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return gamma-gamma*(1-f_aux/gamma)**scalar

    def show_curve(self,lambd=None,beta=None):
        if lambd is None:
            lambd=self.lambd
        if beta is None:
            beta=self.beta
        x=[i/256*self.M for i in range(int(-300/self.M*lambd),int(256/self.M*lambd))]
        x.append(255.9999/256*lambd)
        plt.plot(x, [self.function(i,lambd,beta) for i in x])
        p1=(0,self.function(0,lambd,beta))
        p2=(128/256*self.M,self.function(128/256*self.M,lambd,beta))
        p3=(255/256*self.M,self.function(255/256*self.M,lambd,beta))
        plt.plot(p1[0],p1[1],marker='o',color='red',label=p1)
        plt.plot(p2[0],p2[1],marker='o',color='orange',label=p2)
        plt.plot(p3[0],p3[1],marker='o',color='yellow',label=p3)
        plt.legend()
        plt.title(f'Curva representativa del isomorfismo PLIP, M = {self.M}, λ(M) = {lambd}, β = {beta}')
        plt.xlim(-300/256*lambd,300/256*lambd)
        plt.ylim(-500/256*lambd,2000/256*lambd)
        plt.xlabel('Escala de gris')
        plt.ylabel('φ')
        plt.grid()
        plt.show()