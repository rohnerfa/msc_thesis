from utils.func_yarotsky import get_func_yarotsky
from matplotlib import cm
import matplotlib.pyplot as plt

import numpy as np
import torch

def plot_model_2d(model_, path=None, res=250):
    X = torch.linspace(0,1,res)
    Y = torch.linspace(0,1,res)
    K, L = torch.meshgrid(X, Y, indexing='ij')
    M = torch.cartesian_prod(X, Y)
    M = model_(M).view((res,res)).detach()
    
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(K, L, M, cmap=cm.viridis, linewidth=0, rstride=1, cstride=1, facecolor='skyblue', antialiased=False)
    ax.set_box_aspect((np.ptp(K), np.ptp(L), 2*np.ptp(M)))
    if path is not None:
        plt.savefig(path, dpi=150)
    plt.show()


def plot_function_2d(func_, path=None, res=250):
    X = torch.linspace(0,1,res)
    Y = torch.linspace(0,1,res)
    K, L = torch.meshgrid(X, Y, indexing='ij')
    M = torch.cartesian_prod(X, Y).unsqueeze(-1)
    M = func_(M).view((res,res))
    
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(K, L, M, cmap=cm.viridis, linewidth=0, rstride=1, cstride=1, facecolor='skyblue', antialiased=False)
    ax.set_box_aspect((np.ptp(K), np.ptp(L), 2*np.ptp(M)))
    if path is not None:
        plt.savefig(path, dpi=150)
    plt.show()