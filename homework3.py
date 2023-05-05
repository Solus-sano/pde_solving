import numpy as np
import math as mth
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def init_U_map(J):
    """U的精确解(t==1)"""
    X = np.linspace(0,1,J+1)
    U_map = mth.exp(-mth.pi**2)*np.cos(mth.pi*X)+(1-np.cos(1))
    # U_map = np.zeros((N+1,))
    # for i in range(N+1):
    #     U_map[i] = mth.exp()
    return U_map

def f(x,t):
    return np.exp(-mth.pi**2*t)*np.cos(x*mth.pi)+(1-np.cos(t))


def forward_solving(N,J):
    """前向"""
    h = 1/J;tao = 1/N;r = tao/(h**2)
    U_map = np.zeros((2,J+1))
    U_map[0] = np.cos(mth.pi*np.linspace(0,1,J+1))
    for i in range(1,N+1):
        for j in range(1,J):
            U_map[i%2][j] = r*U_map[(i-1)%2][j-1] + (1-2*r)*U_map[(i-1)%2][j] + r*U_map[(i-1)%2][j+1] + tao*mth.sin(tao*i)
        U_map[i%2][0] = r*U_map[(i-1)%2][1] + (1-2*r)*U_map[(i-1)%2][0] + r*U_map[(i-1)%2][1] + tao*mth.sin(tao*i)
        U_map[i%2][J] = r*U_map[(i-1)%2][J-1] + (1-2*r)*U_map[(i-1)%2][J] + r*U_map[(i-1)%2][J-1] + tao*mth.sin(tao*i)

    return U_map[N%2]

def backward_solving(N,J):
    """后向"""
    h = 1/J;tao = 1/N;r = tao/(h**2)
    U_map = np.zeros((2,J+1))
    U_map[0] = np.cos(mth.pi*np.linspace(0,1,J+1))

    G = np.zeros((J+1,J+1))
    G[0][0] = G[J][J] = 1+2*r
    G[0][1] = G[J][J-1] = -2*r
    for i in range(1,J): G[i][i-1:i+2] = np.array([-r,1+2*r,-r])

    for i in range(1,N+1):
        U_map[i%2] = (np.linalg.inv(G)@((U_map[(i-1)%2]+tao*mth.sin(tao*i)).T)).T


    return U_map[N%2]

def symmetry_solving(N,J):
    """六点对称"""
    h = 1/J;tao = 1/N;r = tao/(h**2)
    U_map = np.zeros((2,J+1))
    U_map[0] = np.cos(mth.pi*np.linspace(0,1,J+1))

    G1 = np.zeros((J+1,J+1))
    G1[0][0] = G1[J][J] = 2+2*r
    G1[0][1] = G1[J][J-1] = -2*r
    for i in range(1,J): G1[i][i-1:i+2] = np.array([-r,2+2*r,-r])

    G2 = np.zeros((J+1,J+1))
    G2[0][0] = G2[J][J] = 2-2*r
    G2[0][1] = G2[J][J-1] = 2*r
    for i in range(1,J): G2[i][i-1:i+2] = np.array([r,2-2*r,r])

    for i in range(1,N+1):
        U_map[i%2] = (np.linalg.inv(G1)@(( G2@U_map[(i-1)%2].T + tao*(mth.sin(tao*i)+mth.sin(tao*(i-1))) ))).T


    return U_map[N%2]

def loss_f(U_lst,init_U_lst):
    """计算两网函数的无穷范数"""
    return np.max(np.abs(U_lst-init_U_lst))

if __name__ == '__main__':
    J = 80; N = 3200 

    print("向前:")
    print("N=3200,J=40: ",loss_f(init_U_map(J=40),forward_solving(N=3200,J=40)))
    print("N=12800,J=80: ",loss_f(init_U_map(J=80),forward_solving(N=12800,J=80)))
    print("向后:")
    print("N=1600,J=40: ",loss_f(init_U_map(J=40),backward_solving(N=1600,J=40)))
    print("N=3200,J=80: ",loss_f(init_U_map(J=80),backward_solving(N=3200,J=80)))
    print("六点对称:")
    print("N=1600,J=40: ",loss_f(init_U_map(J=40),symmetry_solving(N=1600,J=40)))
    print("N=3200,J=80: ",loss_f(init_U_map(J=80),symmetry_solving(N=3200,J=80)))

