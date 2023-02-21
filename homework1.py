import numpy as np
import math as mth
import scipy as sp
import matplotlib.pyplot as plt

def init_U(x):
    """U的精确解"""
    return np.exp(x)*np.sin(x)

def f(x):
    return np.exp(x)*(np.sin(x)-2*np.cos(x))

def q(x):
    return 1

def solving(f,q,a,b,alphe,beta,N):
    """给定方程及边界条件、步长，用有限差分法求解网函数"""
    G=np.zeros((N-1,N-1))
    B=np.zeros((N-1,1))
    X=np.linspace(a,b,N+1)
    h=(b-a)/N

    for i in range(1,N-2):
        G[i][i-1:i+2]=np.array([-1/(h*h),2/(h*h)+q(X[i+1]),-1/(h*h)])
        B[i]=f(X[i+1])
    
    G[0][:2]=np.array([2/(h*h)+q(X[1]),-1/(h*h)])
    G[N-2][-2:]=np.array([-1/(h*h),2/(h*h)+q(X[N-1])])

    B[0],B[N-2]=f(X[1])+alphe/(h*h),f(X[N-1])+beta/(h*h)

    U_lst=np.linalg.inv(G)@B
    U_lst=np.hstack((np.array([alphe]),U_lst.T[0],np.array([beta])))
    return U_lst

def loss_f(U_lst,init_U_lst):
    """计算两网函数的无穷范数"""
    return np.max(np.abs(U_lst-init_U_lst))

def show_solving(N=40):
    """给定步长，可视化求解"""
    a,b=0,mth.pi

    x_lst=np.linspace(a,b,N+1)
    y_init=init_U(np.linspace(a,b,500))
    y_solve=solving(f,q,a,b,init_U(a),init_U(b),N)
    
    plt.plot(np.linspace(a,b,500),y_init)
    plt.scatter(x_lst,y_solve,c='red',s=20)

def show_loss():
    a,b=0,mth.pi
    n_lst=np.arange(10,100,10)
    e_lst=[]
    for n in n_lst:
        x_lst=np.linspace(a,b,n+1)
        y_init=init_U(x_lst)
        y_solve=solving(f,q,a,b,init_U(a),init_U(b),n)
        e_lst.append((loss_f(y_solve,y_init))/((b-a)/n)**2)
        print("N = %d solving finished"%(n))

    plt.figure()
    plt.scatter(n_lst,e_lst)

if __name__ == '__main__':
    show_solving(20)
    show_loss()
    plt.show()