import numpy as np
import math as mth
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def init_U_map(f,N):
    """U的精确解"""
    h1 = mth.pi/N; h2 = 1/N;
    U_map = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            U_map[i][j] = f(i*h1,j*h2)/(9+mth.pi**2)
    return U_map

def f(x,y):
    return np.cos(3*x)*np.sin(mth.pi*y)


def solving(f,N):
    """给定方程及边界条件、步长，用有限差分法求解网函数"""
    h1 = mth.pi/N; h2 = 1/N;

    U_map = np.zeros((N+1,N+1))
    t1 = 1/(h1**2); t2 = 1/(h2**2)
    B = np.zeros((N+1,N+1))
    for i in range(N):
        B[i][i+1] = -t1
        B[i+1][i] = -t1
        B[i+1][i+1] = 2*(t1+t2)
    B[0][0] = 2*(t1+t2); B[0][1] = -2*t1; B[N][-2] = -2*t1

    C = np.diag(np.array([-t2]).repeat(N+1))
    A = np.zeros((N**2-1,N**2-1))
    for i in range(0,N**2-N-2,N+1):
        # print(i)
        # print(A[i+N+1:i+2*N+2,i:i+N+1])
        A[i+N+1:i+2*N+2,i:i+N+1] = C
        A[i:i+N+1,i+N+1:i+2*N+2] = C
        A[i+N+1:i+2*N+2,i+N+1:i+2*N+2] = B
        if i==0: A[i:i+N+1,i:i+N+1] = B

    F = np.zeros((N**2-1,1))
    for i in range(N**2-1): F[i][0] = f((i%(N+1))*h1,(1+i//(N+1))*h2)
    solve_lst = np.linalg.inv(A)@F
    solve_lst = solve_lst.T[0]
    for i in range(N-1):
        for j in range(N+1):
            U_map[j][i+1] = solve_lst[i*(N+1)+j]
        # U_map[:,i+1] = solve_lst[i*(N+1):(i+1)*(N+1)].reshape((1,N+1))

    return U_map

def loss_f(U_lst,init_U_lst):
    """计算两网函数的无穷范数"""
    return np.max(np.abs(U_lst-init_U_lst))

if __name__ == '__main__':
    N=16

    u0 = init_U_map(f,N)
    u = solving(f,N)
    plt.figure()
    ax=plt.axes(projection='3d')
    X,Y = np.meshgrid(np.linspace(0,mth.pi,N+1),np.linspace(0,1,N+1))
    ax.plot_surface(X,Y,u.T)
    # ax.plot_surface(X,Y,u.T)
    print(loss_f(u0,u))
    plt.show()
