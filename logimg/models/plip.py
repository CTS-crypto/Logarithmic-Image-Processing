import math
import numpy as np
import matplotlib.pyplot as plt
from .logimg import LogSpace

class PLIPSpace(LogSpace):
    def __init__(self, M=256,mi=256,gamma=256,k=256,lambd=256,beta=1) -> None:
        super().__init__(M)
        self.mi=mi
        self.gamma=gamma
        self.k=k
        self.lambd=lambd
        self.beta=beta

    def neg(self, f, mi=None):
        if mi is None:
            mi=self.mi
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return mi-f_aux

    def equation(self, f, lambd=None, beta=None):
        if lambd is None:
            lambd=self.lambd
        if beta is None:
            beta=self.beta
        if isinstance(f,np.ndarray):
            return np.array( [ [ -lambd * math.log( 0.0001 if f[i][j]==lambd else 1-f[i][j]/lambd)**beta for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return -lambd * math.log(1.0001-f/lambd if f==lambd else 1-f/lambd)**beta

    def inverse_equation(self, f, lambd=None, beta=None):
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
        return (f_aux-g_aux)/(1+0.0001-g_aux/k)

    def mul(self,f,g,lambd=None,beta=None):
        f_aux=self.equation(f,lambd,beta)
        g_aux=self.equation(g,lambd,beta)
        return self.inverse_equation(f_aux*g_aux,lambd,beta)

    def s_mul(self,f,scalar,gamma=None):
        if gamma is None:
            gamma=self.gamma
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return gamma-gamma*(1-f_aux/gamma)**scalar

    def show_curve(self,lambd=None,beta=None):
        x=range(256)
        plt.plot(x, [self.equation(i,lambd,beta) for i in x])
        plt.title('Curva representativa logarítmica del isomorfismo φ')
        plt.xlim(0,300)
        plt.ylim(0,1600)
        plt.xlabel('gray-scale')
        plt.ylabel('φ')
        plt.grid()
        plt.show()