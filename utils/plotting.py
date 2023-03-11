from utils.func_yarotsky import get_func_yarotsky
from matplotlib import cm
import matplotlib.pyplot as plt

import numpy as np
import torch

def plot_model_2d(model_, path=None):
    res = 200
    # Make data.
    K = np.linspace(0,1,res, dtype=np.float32)
    L = np.linspace(0,1,res, dtype=np.float32)
    K, L = np.meshgrid(K, L)

    samp_ = torch.tensor(np.dstack((K,L)).reshape((-1,2)))
    M_0 = model_(samp_).detach().numpy().reshape(res,res)

    # Plot the surface.
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(projection='3d')

    surf = ax.plot_surface(K, L, M_0, rcount=200, ccount=200, cmap=cm.viridis, linewidth=0, facecolor='skyblue', antialiased=False)
    #ax.set_box_aspect((np.ptp(K), np.ptp(L), np.ptp(M_0)))  # aspect ratio is 1:1:1 in data space
    #ax.grid(False)
    #ax.axis(False)
    if path is not None:
        plt.savefig(path, dpi=150)
    plt.show()



def plot_function_2d(func_, path=None, meshgrid=True):
    N0 = 200
    K = np.linspace(0,1,N0, dtype=np.float32)
    L = np.linspace(0,1,N0, dtype=np.float32)
    K, L = np.meshgrid(K, L)

    if meshgrid:
        M = np.zeros((N0,N0), dtype=np.float32)
        for i in range(N0):
            for j in range(N0):
                M[i,j] = func_(K[i,j], L[i,j])
    
    else:
        M_ = torch.from_numpy(np.stack([K.reshape(-1,), L.reshape(-1,)], axis=1)).unsqueeze(-1)
        M = func_(M_).numpy()
        M = M.reshape((200,200)).T

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(K, L, M, cmap=cm.viridis, linewidth=0, rcount=200, ccount=200, facecolor='skyblue', antialiased=False)
    #ax.set_box_aspect((np.ptp(K), np.ptp(L), np.ptp(M)))  # aspect ratio is 1:1:1 in data space
    if path is not None:
        plt.savefig(path)
    plt.show()