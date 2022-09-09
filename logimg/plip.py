import math
import numpy as np
import matplotlib.pyplot as plt
from logimg import LogImage,LogSpace

class PLIPSpace(LogSpace):
    def __init__(self, M=256) -> None:
        super().__init__(M)

    def neg(self, f, mi):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return mi-1-f_aux

    def equation(self, f, lambd, beta):
        if isinstance(f,np.ndarray):
            return np.array( [ [ -lambd * math.log(1-f[i][j]/lambd)**beta for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return -lambd * math.log(1-f/lambd)**beta

    def inverse_equation(self, f, lambd, beta):
        if isinstance(f,np.ndarray):
            return np.array( [ [ lambd*(1-1/math.e**(f[i][j]/(lambd*beta))) for j in range(f.shape[1])] for i in range(f.shape[0])])
        else:
            return lambd*(1-1/math.e**(f/(lambd*beta)))
     
    def sum(self,f,g,gamma):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist())
        else:
            g_aux=g
        return f_aux+g_aux-(f_aux*g_aux)/gamma

    def sub(self,f,g,k):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        if isinstance(g,np.ndarray):
            g_aux=np.array(g.tolist())
        else:
            g_aux=g
        return (f_aux-g_aux)/(1-g_aux/k)

    def mul(self,f,g,lambd,beta):
        f_aux=self.equation(f,lambd,beta)
        g_aux=self.equation(g,lambd,beta)
        return self.inverse_equation(f_aux*g_aux,lambd,beta)

    def s_mul(self,f,scalar,lambd):
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return lambd-lambd*(1-f_aux/lambd)**scalar

    def show_curve(self,lambd,beta):
        x=range(256)
        plt.plot(x, [self.equation(i,lambd,beta) for i in x])
        plt.title('Curva representativa logarítmica del isomorfismo φ')
        plt.xlim(0,300)
        plt.ylim(0,1600)
        plt.xlabel('gray-scale')
        plt.ylabel('φ')
        plt.grid()
        plt.show()