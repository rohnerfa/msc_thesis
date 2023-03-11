import numpy as np
import torch
torch.manual_seed(42)

def get_func_yarotsky(n_peaks):
    def phi(x_,y_):
        x = 4*x_
        y = 4*y_
        z = np.sqrt(x**2 + y**2)
        return np.exp(-1/(1-(x**2+y**2)*(z<1)))*np.e * (z < 1)

    N = n_peaks
    X1 = np.linspace(0,1,N+2, dtype=np.float32)
    X1 = X1[1:-1]
    X2 = np.linspace(0,1,N+2, dtype=np.float32)
    X2 = X2[1:-1]
    X1, X2 = np.meshgrid(X1, X2)
    
    np.random.seed(42)
    y = 0.25*np.random.randint(2, size=(N,N))

    def f(x1,x2):
    
        z = np.multiply(phi(N*(x1-X1),N*(x2-X2)),y)
        return np.sum(z)

    return f

def get_func(n_peaks, seed=42):
    
    def phi2(T):
        if T.dim() != 3:
            raise ValueError("wrong input dimension, expected: (None,2,1)")
        T = 4*T
        z = torch.sqrt(torch.pow(T[:,0,:],2) + torch.pow(T[:,1,:],2))
        return torch.exp(-1/(1-(torch.pow(T[:,0,:],2) + torch.pow(T[:,1,:],2)))*(z<1))*torch.e*(z<1)

    def func(K_):
        N = n_peaks
        X1 = torch.linspace(0,1,N+2)
        X1 = X1[1:-1]
        X2 = torch.linspace(0,1,N+2)
        X2 = X2[1:-1]
        X = torch.cartesian_prod(X1,X2).T

        torch.manual_seed(seed)
        y = 0.25*torch.randint(2, size=(N**2,))

        return torch.matmul(phi2(N*(K_ - X)), y)
    return func