from .pslip import PSLIPSpace
import numpy as np

class PPSLIPSpace(PSLIPSpace):
    def gray_tone(self,f,M=None):
        if M is None:
            return super().gray_tone(f)
        if isinstance(f,np.ndarray):
            f_aux=np.array(f.tolist())
        else:
            f_aux=f
        return f_aux/M

    def inverse_gray_tone(self,f,M=None):
        if M is None:
            return super().gray_tone(f)
        else:
            return M*f