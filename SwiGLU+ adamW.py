
import torch.nn as nn
import torch.nn.functional as F
import math 


class SwiGlu(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.W1 = nn.Linear(d_model, d_ff, bias = False)
        self.W2 = nn.Linear(d_model, d_ff,bias = False)
        self.W3 = nn.Linear(d_ff, d_model,bias = False)


    def forward(self, x):
        # x: [B,T,d_model], W1(x), W2(x): [B,T,d_ff] 
        x = self.W3(F.silu(self.W1(x))*self.W2(x))
        return x 



def adam_step(param, grad, m,v,t,lr = 1e-3 , beta1 = 0.9, beta2 = 0.99, eps = 1e-8, weight_decay = 0.0 ):
    if weight_decay != 0:
        grad = grad + weight_decay* param

    m = beta1* m + (1-beta1)*grad
    v = beta2* v + (1-beta2)*grad.pow(2)
    m_hat = m/(1-beta1**t) # bias-corrected estimates; corect the underestimate of m and v at time 0 
    v_hat = v/(1-beta2**t)

    param = param - lr*m_hat/(torch.sqrt(v_hat)+eps)

    return param, m, v

def adamW_step(param, grad, m,v,t,lr = 1e-3 , beta1 = 0.9, beta2 = 0.99, eps = 1e-8, weight_decay = 0.0 ):
    
    m = beta1* m + (1-beta1)*grad
    v = beta2* v + (1-beta2)*grad.pow(2)
    m_hat = m/(1-beta1**t)
    v_hat = v/(1-beta2**t)

    param = param - lr*m_hat/(torch.sqrt(v_hat)+eps)
    param = param - lr*weight_decay*param

    return param, m, v
    



